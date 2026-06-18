// JNI wrapper around llama.cpp's C API (llama.h).
// Mirrors the single-sequence usage pattern from llama.cpp's examples/simple/simple.cpp.

#include "dev_localllm_jni_LlamaNative.h"

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

} // namespace

namespace {
void quietLogCallback(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}
} // namespace

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_backendInit
  (JNIEnv*, jclass) {
    llama_log_set(quietLogCallback, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
}

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_backendFree
  (JNIEnv*, jclass) {
    llama_backend_free();
}

JNIEXPORT jlong JNICALL Java_dev_localllm_jni_LlamaNative_loadModel
  (JNIEnv* env, jclass, jstring jpath, jint nGpuLayers) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = nGpuLayers;

    llama_model* model = llama_model_load_from_file(path, params);

    env->ReleaseStringUTFChars(jpath, path);

    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_freeModel
  (JNIEnv*, jclass, jlong handle) {
    if (handle != 0) {
        llama_model_free(asModel(handle));
    }
}

JNIEXPORT jlong JNICALL Java_dev_localllm_jni_LlamaNative_newContext
  (JNIEnv*, jclass, jlong modelHandle, jint nCtx, jint nThreads) {
    llama_model* model = asModel(modelHandle);

    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(nCtx);
    params.n_batch = static_cast<uint32_t>(nCtx);
    params.n_threads = nThreads;
    params.n_threads_batch = nThreads;

    llama_context* ctx = llama_init_from_model(model, params);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL Java_dev_localllm_jni_LlamaNative_freeContext
  (JNIEnv*, jclass, jlong handle) {
    if (handle != 0) {
        llama_free(asContext(handle));
    }
}

JNIEXPORT jintArray JNICALL Java_dev_localllm_jni_LlamaNative_tokenize
  (JNIEnv* env, jclass, jlong modelHandle, jstring jtext, jboolean addSpecial, jboolean parseSpecial) {
    const llama_vocab* vocab = llama_model_get_vocab(asModel(modelHandle));

    const char* text = env->GetStringUTFChars(jtext, nullptr);
    jsize textLen = env->GetStringUTFLength(jtext);

    int nTokens = -llama_tokenize(vocab, text, textLen, nullptr, 0, addSpecial, parseSpecial);

    std::vector<llama_token> tokens(nTokens > 0 ? nTokens : 0);
    if (nTokens > 0) {
        int written = llama_tokenize(vocab, text, textLen, tokens.data(), nTokens, addSpecial, parseSpecial);
        if (written < 0) {
            tokens.clear();
        }
    }

    env->ReleaseStringUTFChars(jtext, text);

    jintArray result = env->NewIntArray(static_cast<jsize>(tokens.size()));
    if (!tokens.empty()) {
        static_assert(sizeof(llama_token) == sizeof(jint), "llama_token must be 32-bit");
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(tokens.size()),
                                reinterpret_cast<const jint*>(tokens.data()));
    }
    return result;
}

JNIEXPORT jstring JNICALL Java_dev_localllm_jni_LlamaNative_tokenToPiece
  (JNIEnv* env, jclass, jlong modelHandle, jint token) {
    const llama_vocab* vocab = llama_model_get_vocab(asModel(modelHandle));
    std::string piece = tokenToPieceStr(vocab, token);
    return env->NewStringUTF(piece.c_str());
}

JNIEXPORT jboolean JNICALL Java_dev_localllm_jni_LlamaNative_isEog
  (JNIEnv*, jclass, jlong modelHandle, jint token) {
    const llama_vocab* vocab = llama_model_get_vocab(asModel(modelHandle));
    return llama_vocab_is_eog(vocab, token) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_dev_localllm_jni_LlamaNative_generate
  (JNIEnv* env, jclass, jlong modelHandle, jlong ctxHandle, jintArray jpromptTokens,
   jint nPredict, jfloat temperature, jobject callback) {

    llama_model* model = asModel(modelHandle);
    llama_context* ctx = asContext(ctxHandle);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    jsize nPrompt = env->GetArrayLength(jpromptTokens);
    std::vector<llama_token> tokens(nPrompt);
    {
        jint* raw = env->GetIntArrayElements(jpromptTokens, nullptr);
        std::memcpy(tokens.data(), raw, nPrompt * sizeof(jint));
        env->ReleaseIntArrayElements(jpromptTokens, raw, JNI_ABORT);
    }

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)Z");

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
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
        jboolean keepGoing = env->CallBooleanMethod(callback, onToken, jpiece);
        env->DeleteLocalRef(jpiece);

        nDecoded++;

        if (!keepGoing) {
            break;
        }

        batch = llama_batch_get_one(&newToken, 1);
    }

    llama_sampler_free(smpl);

    return nDecoded;
}
