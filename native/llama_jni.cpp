// JNI wrapper around llama.cpp's C API (llama.h).
// Mirrors the single-sequence usage pattern from llama.cpp's examples/simple/simple.cpp.
//
// Crash containment
// ------------------
// Without precautions, two classes of native failure take the whole JVM
// process down instead of surfacing as a catchable Java exception:
//
//   1. A C++ exception (e.g. std::bad_alloc) escaping across the JNI
//      boundary - this is undefined behavior per the JNI spec.
//   2. A fatal signal (SIGSEGV from a bad pointer, SIGABRT from a failed
//      GGML_ASSERT, SIGBUS, SIGFPE, SIGILL) raised inside llama.cpp/ggml.
//
// Every exported function therefore: validates its arguments before
// touching native memory (most realistic crash causes are simply bad
// input), wraps its body in try/catch, and runs inside a per-call signal
// guard (see LLAMA_JNI_GUARD_BEGIN/END below) that converts a caught
// signal into a NativeCrashException.
//
// Important limits on what this can guarantee:
//   - A signal caught mid-llama_decode means native heap state may be
//     corrupted in ways invisible from Java. We do not pretend the
//     process can keep using llama.cpp safely afterward: g_crashed latches
//     permanently and every further native call refuses to run. The
//     process should be restarted, not retried in place.
//   - The signal guard is installed per-thread (thread_local) and only
//     covers the thread that called into JNI. ggml's internal worker
//     threads (used for batched decode with n_threads > 1) are not
//     wrapped individually; a crash on one of those threads is not
//     caught and will still terminate the process. This is a known gap,
//     not a guarantee we can close without instrumenting ggml itself.
//   - Mixing sigsetjmp/siglongjmp with C++ objects that have non-trivial
//     destructors is technically UB (destructors between the setjmp and
//     a later siglongjmp do not run). We accept best-effort leaks on the
//     crash path in exchange for not losing the whole process - the
//     guard is the first statement in every function, before any such
//     object is constructed, so this only affects the (already
//     exceptional) crash path.

#include "dev_localllm_jni_LlamaNative.h"

#include "llama.h"
#include "ggml-backend.h"

#include <atomic>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

inline llama_model* asModel(jlong handle) {
    return reinterpret_cast<llama_model*>(handle);
}

inline llama_context* asContext(jlong handle) {
    return reinterpret_cast<llama_context*>(handle);
}

// Converts a token to its text piece. Returns empty string on failure.
std::string tokenToPieceStr(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return std::string();
    }
    return std::string(buf, n);
}

// ---------------------------------------------------------------------
// Log forwarding into Java (see setLogCallback / forwardLogCallback below,
// defined after the JNI exception helpers since they reuse them).
// ---------------------------------------------------------------------

// The Java callback registered via setLogCallback(), or nullptr if none.
// Set once at startup (see NativeLogBridge.installOnce() on the Java
// side) before any concurrent model/context use begins; forwardLogCallback
// only ever reads it. We still publish it through an atomic release/
// acquire pair so a reader on another thread is guaranteed to see the
// fully-initialized jmethodID written just before it (see setLogCallback).
std::atomic<jobject> g_logCallbackRef{nullptr};
jmethodID g_logMethodId = nullptr;

// JNIEnv for the JNI call currently in flight on this thread, set by
// GuardActiveScope. Native log messages are only ever observed to be
// emitted synchronously within a guarded JNI call (model load, context
// creation, decode), so this is sufficient to call back into Java
// without the overhead/complexity of JavaVM::AttachCurrentThread.
thread_local JNIEnv* g_currentEnv = nullptr;

// ---------------------------------------------------------------------
// Java exception helpers
// ---------------------------------------------------------------------

void throwJavaException(JNIEnv* env, const char* className, const std::string& message) {
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
    jclass cls = env->FindClass(className);
    if (cls != nullptr) {
        env->ThrowNew(cls, message.c_str());
        env->DeleteLocalRef(cls);
    }
}

void throwIllegalArgument(JNIEnv* env, const std::string& msg) {
    throwJavaException(env, "java/lang/IllegalArgumentException", msg);
}

void throwIllegalState(JNIEnv* env, const std::string& msg) {
    throwJavaException(env, "java/lang/IllegalStateException", msg);
}

void throwOutOfMemory(JNIEnv* env, const std::string& msg) {
    throwJavaException(env, "java/lang/OutOfMemoryError", msg);
}

void throwNativeCrash(JNIEnv* env, const std::string& detail) {
    std::string msg =
        "native code crashed (" + detail + "); the llama.cpp runtime may be in a "
        "corrupted state and will refuse further calls in this process - restart "
        "the process before retrying.";
    throwJavaException(env, "dev/localllm/jni/NativeCrashException", msg);
}

// ---------------------------------------------------------------------
// RAII helpers for JNI string/array pins, so every early-return /
// thrown-exception path on the *normal* (non-crash) control flow still
// releases them via stack unwinding.
// ---------------------------------------------------------------------

class JStringUTF {
public:
    JStringUTF(JNIEnv* env, jstring str) : env_(env), str_(str), chars_(nullptr) {
        if (str_ != nullptr) {
            chars_ = env_->GetStringUTFChars(str_, nullptr);
        }
    }
    ~JStringUTF() {
        if (chars_ != nullptr) {
            env_->ReleaseStringUTFChars(str_, chars_);
        }
    }
    JStringUTF(const JStringUTF&) = delete;
    JStringUTF& operator=(const JStringUTF&) = delete;

    const char* get() const { return chars_; }

private:
    JNIEnv* env_;
    jstring str_;
    const char* chars_;
};

class JIntArrayElems {
public:
    JIntArrayElems(JNIEnv* env, jintArray arr) : env_(env), arr_(arr), elems_(nullptr) {
        if (arr_ != nullptr) {
            elems_ = env_->GetIntArrayElements(arr_, nullptr);
        }
    }
    ~JIntArrayElems() {
        if (elems_ != nullptr) {
            env_->ReleaseIntArrayElements(arr_, elems_, JNI_ABORT);
        }
    }
    JIntArrayElems(const JIntArrayElems&) = delete;
    JIntArrayElems& operator=(const JIntArrayElems&) = delete;

    jint* get() const { return elems_; }

private:
    JNIEnv* env_;
    jintArray arr_;
    jint* elems_;
};

// ---------------------------------------------------------------------
// Signal handling / crash containment
// ---------------------------------------------------------------------

// Set once any guarded call catches a fatal signal. After this, every
// JNI entry point refuses to touch native memory again.
std::atomic<bool> g_crashed{false};

thread_local sigjmp_buf g_jumpEnv;
thread_local volatile sig_atomic_t g_guardActive = 0;
thread_local std::unique_ptr<char[]> g_altStack;

struct sigaction g_prevSegv{};
struct sigaction g_prevAbrt{};
struct sigaction g_prevBus{};
struct sigaction g_prevFpe{};
struct sigaction g_prevIll{};
std::atomic<bool> g_handlersInstalled{false};

const char* signalName(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV: segmentation fault";
        case SIGABRT: return "SIGABRT: abort (e.g. a failed GGML_ASSERT)";
        case SIGBUS:  return "SIGBUS: bus error";
        case SIGFPE:  return "SIGFPE: floating point exception";
        case SIGILL:  return "SIGILL: illegal instruction";
        default:      return "unknown fatal signal";
    }
}

struct sigaction* prevHandlerFor(int sig) {
    switch (sig) {
        case SIGSEGV: return &g_prevSegv;
        case SIGABRT: return &g_prevAbrt;
        case SIGBUS:  return &g_prevBus;
        case SIGFPE:  return &g_prevFpe;
        case SIGILL:  return &g_prevIll;
        default:      return nullptr;
    }
}

// Async-signal-safe best-effort diagnostic write (no fprintf/locking).
void logSignalAsyncSafe(int sig) {
    const char* name = signalName(sig);
    static const char prefix[] = "[llamajni] caught fatal signal: ";
    static const char suffix[] = "\n";
    ssize_t ignored;
    ignored = write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    ignored = write(STDERR_FILENO, name, strlen(name));
    ignored = write(STDERR_FILENO, suffix, sizeof(suffix) - 1);
    (void) ignored;
}

void crashSignalHandler(int sig, siginfo_t* info, void* ucontext) {
    if (g_guardActive) {
        logSignalAsyncSafe(sig);
        siglongjmp(g_jumpEnv, sig);
        // unreachable
    }

    // Not inside a guarded JNI call (e.g. a crash on a ggml worker thread
    // we don't wrap, or a signal the JVM itself relies on for unrelated
    // purposes). Chain to whatever was previously installed so we don't
    // hijack behavior the JVM depends on; if there's nothing sensible to
    // chain to, restore default disposition and re-raise so the process
    // still terminates normally (e.g. with a core dump) instead of
    // spinning in a broken handler.
    struct sigaction* prev = prevHandlerFor(sig);
    if (prev != nullptr && (prev->sa_flags & SA_SIGINFO) && prev->sa_sigaction != nullptr) {
        prev->sa_sigaction(sig, info, ucontext);
        return;
    }
    if (prev != nullptr && prev->sa_handler != nullptr &&
        prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
        prev->sa_handler(sig);
        return;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

void ensureAltStackInstalled() {
    if (g_altStack) {
        return;
    }
    constexpr size_t kAltStackSize = 131072; // 128 KiB, comfortably above MINSIGSTKSZ
    g_altStack.reset(new char[kAltStackSize]);
    stack_t ss;
    ss.ss_sp = g_altStack.get();
    ss.ss_size = kAltStackSize;
    ss.ss_flags = 0;
    sigaltstack(&ss, nullptr);
}

void installCrashHandlersOnce() {
    if (g_handlersInstalled.exchange(true)) {
        return;
    }
    struct sigaction act{};
    sigemptyset(&act.sa_mask);
    act.sa_sigaction = crashSignalHandler;
    // SA_ONSTACK: run on the alternate stack so a SIGSEGV caused by stack
    // overflow can still be handled instead of immediately re-faulting.
    // No SA_NODEFER: leave the signal blocked for the duration of the
    // handler: siglongjmp() restores the mask saved at sigsetjmp() time,
    // which is what correctly unblocks it again afterward.
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;

    sigaction(SIGSEGV, &act, &g_prevSegv);
    sigaction(SIGABRT, &act, &g_prevAbrt);
    sigaction(SIGBUS,  &act, &g_prevBus);
    sigaction(SIGFPE,  &act, &g_prevFpe);
    sigaction(SIGILL,  &act, &g_prevIll);
}

// RAII flip of g_guardActive for the *normal* control flow: its
// destructor runs on every ordinary return/throw via stack unwinding. On
// the signal-recovery path (siglongjmp), destructors are skipped, so
// LLAMA_JNI_GUARD_BEGIN's recovery branch resets the flag manually there.
// Also stashes the current JNIEnv* (see g_currentEnv) for the duration of
// the call, restoring whatever was there before on the way out (JNI calls
// can re-enter, e.g. a Java callback that itself calls back into native).
struct GuardActiveScope {
    JNIEnv* prevEnv;
    explicit GuardActiveScope(JNIEnv* env) : prevEnv(g_currentEnv) {
        g_guardActive = 1;
        g_currentEnv = env;
    }
    ~GuardActiveScope() {
        g_guardActive = 0;
        g_currentEnv = prevEnv;
    }
};

// Buffers native log output per-thread and flushes it to Java one
// complete line at a time. llama.cpp/ggml emit GGML_LOG_LEVEL_CONT for
// "continuation" output (e.g. the "...." progress dots printed while
// loading tensors) that has no level of its own and no line break until
// the caller is done - forwarding each fragment as its own log call would
// flood the Java-side logger with single-character lines, so we coalesce
// on '\n' instead and tag the merged line with the level of the message
// that started it.
thread_local std::string g_logLineBuffer;
thread_local ggml_log_level g_logLineLevel = GGML_LOG_LEVEL_INFO;

void emitLogLine(const std::string& line, ggml_log_level level) {
    if (line.empty()) {
        return;
    }

    JNIEnv* env = g_currentEnv;
    jobject cb = g_logCallbackRef.load(std::memory_order_acquire);
    if (env == nullptr || cb == nullptr) {
        // No Java bridge registered yet (or this log line was produced
        // off a thread we don't track a JNIEnv for) - fall back to stderr
        // for warnings/errors rather than losing them silently.
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(line.c_str(), stderr);
            fputc('\n', stderr);
        }
        return;
    }

    jstring jline = env->NewStringUTF(line.c_str());
    if (jline == nullptr) {
        env->ExceptionClear(); // OOM constructing the string - drop this line
        return;
    }
    env->CallVoidMethod(cb, g_logMethodId, static_cast<jint>(level), jline);
    if (env->ExceptionCheck()) {
        // Don't let a misbehaving Java log listener corrupt native control
        // flow; the exception would otherwise surface at an unrelated JNI
        // call site.
        env->ExceptionClear();
    }
    env->DeleteLocalRef(jline);
}

void forwardLogCallback(ggml_log_level level, const char* text, void*) {
    if (text == nullptr) {
        return;
    }
    if (level != GGML_LOG_LEVEL_CONT) {
        g_logLineLevel = level;
    }
    g_logLineBuffer.append(text);

    size_t newlinePos;
    while ((newlinePos = g_logLineBuffer.find('\n')) != std::string::npos) {
        emitLogLine(g_logLineBuffer.substr(0, newlinePos), g_logLineLevel);
        g_logLineBuffer.erase(0, newlinePos + 1);
    }
}

} // namespace

// Establishes the per-call crash guard. Must be the first statement(s)
// in a JNI function, before any non-trivial C++ local is constructed.
// `sentinel` is the value returned on the (rare) crash / already-crashed
// path; use `;` for void-returning functions.
#define LLAMA_JNI_GUARD_BEGIN(env, sentinel)                                   \
    if (g_crashed.load(std::memory_order_acquire)) {                          \
        throwNativeCrash((env), "native runtime already crashed previously"); \
        return sentinel;                                                      \
    }                                                                         \
    ensureAltStackInstalled();                                                \
    int llamajni_sig_ = sigsetjmp(g_jumpEnv, 1);                            \
    if (llamajni_sig_ != 0) {                                               \
        g_guardActive = 0;                                                    \
        g_crashed.store(true, std::memory_order_release);                     \
        throwNativeCrash((env), signalName(llamajni_sig_));                 \
        return sentinel;                                                      \
    }                                                                         \
    GuardActiveScope llamajni_guard_scope_(env)

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_backendInit
  (JNIEnv* env, jclass) {
    LLAMA_JNI_GUARD_BEGIN(env, );
    try {
        installCrashHandlersOnce();
        llama_log_set(forwardLogCallback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("backendInit failed: ") + e.what());
    } catch (...) {
        throwIllegalState(env, "backendInit failed: unknown native exception");
    }
}

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_backendFree
  (JNIEnv* env, jclass) {
    LLAMA_JNI_GUARD_BEGIN(env, );
    try {
        llama_backend_free();
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("backendFree failed: ") + e.what());
    } catch (...) {
        throwIllegalState(env, "backendFree failed: unknown native exception");
    }
}

// Registers (or, with callback == null, clears) the Java listener that
// forwardLogCallback forwards llama.cpp/ggml log output to. Call this
// before backendInit() so hardware-detection / model-load logging is
// captured from the very first message instead of falling back to stderr.
JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_setLogCallback
  (JNIEnv* env, jclass, jobject callback) {
    LLAMA_JNI_GUARD_BEGIN(env, );

    jobject oldRef = g_logCallbackRef.exchange(nullptr, std::memory_order_acq_rel);
    if (oldRef != nullptr) {
        env->DeleteGlobalRef(oldRef);
    }
    g_logMethodId = nullptr;

    if (callback == nullptr) {
        return;
    }

    jclass cbClass = env->GetObjectClass(callback);
    jmethodID mid = env->GetMethodID(cbClass, "onLog", "(ILjava/lang/String;)V");
    env->DeleteLocalRef(cbClass);
    if (mid == nullptr) {
        // GetMethodID already left a pending exception (NoSuchMethodError).
        return;
    }

    jobject newRef = env->NewGlobalRef(callback);
    if (newRef == nullptr) {
        throwOutOfMemory(env, "failed to create global ref for log callback");
        return;
    }

    // Publish the jmethodID before the ref so an acquire-load of
    // g_logCallbackRef on another thread is guaranteed to see it too.
    g_logMethodId = mid;
    g_logCallbackRef.store(newRef, std::memory_order_release);
}

JNIEXPORT jlong JNICALL Java_dev_localllm_jni_LlamaNative_loadModel
  (JNIEnv* env, jclass, jstring jpath, jint nGpuLayers) {
    LLAMA_JNI_GUARD_BEGIN(env, 0);

    if (jpath == nullptr) {
        throwIllegalArgument(env, "path must not be null");
        return 0;
    }

    JStringUTF path(env, jpath);
    if (path.get() == nullptr) {
        throwOutOfMemory(env, "failed to pin model path string");
        return 0;
    }

    try {
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = nGpuLayers;

        llama_model* model = llama_model_load_from_file(path.get(), params);
        return reinterpret_cast<jlong>(model);
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("loadModel failed: ") + e.what());
        return 0;
    } catch (...) {
        throwIllegalState(env, "loadModel failed: unknown native exception");
        return 0;
    }
}

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_freeModel
  (JNIEnv* env, jclass, jlong handle) {
    LLAMA_JNI_GUARD_BEGIN(env, );
    try {
        if (handle != 0) {
            llama_model_free(asModel(handle));
        }
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("freeModel failed: ") + e.what());
    } catch (...) {
        throwIllegalState(env, "freeModel failed: unknown native exception");
    }
}

JNIEXPORT jlong JNICALL Java_dev_localllm_jni_LlamaNative_newContext
  (JNIEnv* env, jclass, jlong modelHandle, jint nCtx, jint nThreads) {
    LLAMA_JNI_GUARD_BEGIN(env, 0);

    if (modelHandle == 0) {
        throwIllegalArgument(env, "model handle must not be 0 (model not loaded or already freed)");
        return 0;
    }
    if (nCtx <= 0) {
        throwIllegalArgument(env, "nCtx must be positive");
        return 0;
    }
    if (nThreads <= 0) {
        throwIllegalArgument(env, "nThreads must be positive");
        return 0;
    }

    try {
        llama_model* model = asModel(modelHandle);

        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(nCtx);
        params.n_batch = static_cast<uint32_t>(nCtx);
        params.n_threads = nThreads;
        params.n_threads_batch = nThreads;

        llama_context* ctx = llama_init_from_model(model, params);
        return reinterpret_cast<jlong>(ctx);
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("newContext failed: ") + e.what());
        return 0;
    } catch (...) {
        throwIllegalState(env, "newContext failed: unknown native exception");
        return 0;
    }
}

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_freeContext
  (JNIEnv* env, jclass, jlong handle) {
    LLAMA_JNI_GUARD_BEGIN(env, );
    try {
        if (handle != 0) {
            llama_free(asContext(handle));
        }
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("freeContext failed: ") + e.what());
    } catch (...) {
        throwIllegalState(env, "freeContext failed: unknown native exception");
    }
}

JNIEXPORT jintArray JNICALL Java_dev_localllm_jni_LlamaNative_tokenize
  (JNIEnv* env, jclass, jlong modelHandle, jstring jtext, jboolean addSpecial, jboolean parseSpecial) {
    LLAMA_JNI_GUARD_BEGIN(env, nullptr);

    if (modelHandle == 0) {
        throwIllegalArgument(env, "model handle must not be 0 (model not loaded or already freed)");
        return nullptr;
    }
    if (jtext == nullptr) {
        throwIllegalArgument(env, "text must not be null");
        return nullptr;
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(asModel(modelHandle));
        if (vocab == nullptr) {
            throwIllegalState(env, "tokenize failed: model has no vocab");
            return nullptr;
        }

        JStringUTF text(env, jtext);
        if (text.get() == nullptr) {
            throwOutOfMemory(env, "failed to pin text string");
            return nullptr;
        }
        jsize textLen = env->GetStringUTFLength(jtext);

        int nTokens = -llama_tokenize(vocab, text.get(), textLen, nullptr, 0, addSpecial, parseSpecial);

        std::vector<llama_token> tokens(nTokens > 0 ? nTokens : 0);
        if (nTokens > 0) {
            int written = llama_tokenize(vocab, text.get(), textLen, tokens.data(), nTokens, addSpecial, parseSpecial);
            if (written < 0) {
                tokens.clear();
            }
        }

        jintArray result = env->NewIntArray(static_cast<jsize>(tokens.size()));
        if (result == nullptr) {
            // NewIntArray already threw OutOfMemoryError.
            return nullptr;
        }
        if (!tokens.empty()) {
            static_assert(sizeof(llama_token) == sizeof(jint), "llama_token must be 32-bit");
            env->SetIntArrayRegion(result, 0, static_cast<jsize>(tokens.size()),
                                    reinterpret_cast<const jint*>(tokens.data()));
        }
        return result;
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("tokenize failed: ") + e.what());
        return nullptr;
    } catch (...) {
        throwIllegalState(env, "tokenize failed: unknown native exception");
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL Java_dev_localllm_jni_LlamaNative_tokenToPiece
  (JNIEnv* env, jclass, jlong modelHandle, jint token) {
    LLAMA_JNI_GUARD_BEGIN(env, nullptr);

    if (modelHandle == 0) {
        throwIllegalArgument(env, "model handle must not be 0 (model not loaded or already freed)");
        return nullptr;
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(asModel(modelHandle));
        if (vocab == nullptr) {
            throwIllegalState(env, "tokenToPiece failed: model has no vocab");
            return nullptr;
        }
        std::string piece = tokenToPieceStr(vocab, token);
        return env->NewStringUTF(piece.c_str());
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("tokenToPiece failed: ") + e.what());
        return nullptr;
    } catch (...) {
        throwIllegalState(env, "tokenToPiece failed: unknown native exception");
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL Java_dev_localllm_jni_LlamaNative_isEog
  (JNIEnv* env, jclass, jlong modelHandle, jint token) {
    LLAMA_JNI_GUARD_BEGIN(env, JNI_FALSE);

    if (modelHandle == 0) {
        throwIllegalArgument(env, "model handle must not be 0 (model not loaded or already freed)");
        return JNI_FALSE;
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(asModel(modelHandle));
        if (vocab == nullptr) {
            throwIllegalState(env, "isEog failed: model has no vocab");
            return JNI_FALSE;
        }
        return llama_vocab_is_eog(vocab, token) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("isEog failed: ") + e.what());
        return JNI_FALSE;
    } catch (...) {
        throwIllegalState(env, "isEog failed: unknown native exception");
        return JNI_FALSE;
    }
}

JNIEXPORT jint JNICALL Java_dev_localllm_jni_LlamaNative_generate
  (JNIEnv* env, jclass, jlong modelHandle, jlong ctxHandle, jintArray jpromptTokens,
   jint nPredict, jfloat temperature, jobject callback) {
    LLAMA_JNI_GUARD_BEGIN(env, 0);

    if (modelHandle == 0) {
        throwIllegalArgument(env, "model handle must not be 0 (model not loaded or already freed)");
        return 0;
    }
    if (ctxHandle == 0) {
        throwIllegalArgument(env, "context handle must not be 0 (context not created or already freed)");
        return 0;
    }
    if (jpromptTokens == nullptr) {
        throwIllegalArgument(env, "promptTokens must not be null");
        return 0;
    }
    if (env->GetArrayLength(jpromptTokens) <= 0) {
        throwIllegalArgument(env, "promptTokens must not be empty");
        return 0;
    }
    if (nPredict < 0) {
        throwIllegalArgument(env, "nPredict must not be negative");
        return 0;
    }
    if (callback == nullptr) {
        throwIllegalArgument(env, "callback must not be null");
        return 0;
    }

    try {
        llama_model* model = asModel(modelHandle);
        llama_context* ctx = asContext(ctxHandle);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            throwIllegalState(env, "generate failed: model has no vocab");
            return 0;
        }

        jsize nPrompt = env->GetArrayLength(jpromptTokens);
        std::vector<llama_token> tokens(nPrompt);
        {
            JIntArrayElems raw(env, jpromptTokens);
            if (raw.get() == nullptr) {
                throwOutOfMemory(env, "failed to pin promptTokens array");
                return 0;
            }
            std::memcpy(tokens.data(), raw.get(), nPrompt * sizeof(jint));
        }

        jclass callbackClass = env->GetObjectClass(callback);
        jmethodID onToken = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)Z");
        if (onToken == nullptr) {
            // GetMethodID already left a pending exception (NoSuchMethodError).
            return 0;
        }

        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler* smpl = llama_sampler_chain_init(sparams);
        if (smpl == nullptr) {
            throwIllegalState(env, "generate failed: could not initialize sampler chain");
            return 0;
        }
        if (temperature <= 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }

        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));

        int nDecoded = 0;
        llama_token newToken;

        while (nDecoded < nPredict) {
            if (llama_decode(ctx, batch) != 0) {
                break;
            }

            newToken = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, newToken)) {
                break;
            }

            std::string piece = tokenToPieceStr(vocab, newToken);
            jstring jpiece = env->NewStringUTF(piece.c_str());
            if (jpiece == nullptr) {
                break; // OutOfMemoryError already pending
            }
            jboolean keepGoing = env->CallBooleanMethod(callback, onToken, jpiece);
            env->DeleteLocalRef(jpiece);

            nDecoded++;

            if (env->ExceptionCheck()) {
                // The Java callback threw - stop generating and let it propagate.
                break;
            }
            if (!keepGoing) {
                break;
            }

            batch = llama_batch_get_one(&newToken, 1);
        }

        llama_sampler_free(smpl);

        return nDecoded;
    } catch (const std::exception& e) {
        throwIllegalState(env, std::string("generate failed: ") + e.what());
        return 0;
    } catch (...) {
        throwIllegalState(env, "generate failed: unknown native exception");
        return 0;
    }
}
