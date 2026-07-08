package dev.localllm.pull;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiConsumer;

/**
 * Minimal HuggingFace Hub client: list GGUF files in a repo and stream-download one.
 *
 * <p>Authentication: pass a HF access token to the constructor (or {@code null} for
 * public repos). Tokens can be created at https://huggingface.co/settings/tokens.
 *
 * <p>Download URL pattern:
 * {@code https://huggingface.co/{owner}/{repo}/resolve/{branch}/{filePath}}
 * HuggingFace redirects to CDN; the built-in {@link HttpClient} follows redirects.
 */
public class HuggingFaceClient {

    private static final String HF_BASE = "https://huggingface.co";
    private static final String HF_API  = "https://huggingface.co/api";

    public static final class HfFile {
        public final String name;
        HfFile(String name) { this.name = name; }
    }

    private final String     token;
    private final HttpClient http;

    public HuggingFaceClient(String token) {
        this.token = token;
        this.http  = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    /** Returns all {@code .gguf} file paths listed in the repo, sorted alphabetically. */
    public List<HfFile> listGgufFiles(String owner, String repo) throws Exception {
        String url  = HF_API + "/models/" + owner + "/" + repo;
        HttpResponse<String> resp = http.send(
                authed(HttpRequest.newBuilder(URI.create(url)).GET()
                        .timeout(Duration.ofSeconds(30))).build(),
                HttpResponse.BodyHandlers.ofString());
        checkStatus(resp.statusCode(), resp.body());

        JsonObject obj = JsonParser.parseString(resp.body()).getAsJsonObject();
        JsonArray siblings = obj.getAsJsonArray("siblings");
        List<HfFile> files = new ArrayList<>();
        if (siblings != null) {
            for (JsonElement el : siblings) {
                String fname = el.getAsJsonObject().get("rfilename").getAsString();
                if (fname.toLowerCase().endsWith(".gguf")) {
                    files.add(new HfFile(fname));
                }
            }
        }
        files.sort(Comparator.comparing(f -> f.name));
        return files;
    }

    /**
     * Stream-download {@code filePath} from {@code owner/repo} at {@code branch} into {@code dest}.
     *
     * <p>A {@code dest + ".part"} temp file is used during transfer and atomically renamed on
     * success. The temp file is deleted on error so callers can safely retry.
     *
     * @param onProgress called after each buffer flush with (bytesDownloaded, totalBytes);
     *                   totalBytes is -1 when the server does not send Content-Length
     */
    public void download(String owner, String repo, String filePath, String branch,
                         Path dest, BiConsumer<Long, Long> onProgress) throws Exception {
        String url = HF_BASE + "/" + owner + "/" + repo + "/resolve/" + branch + "/" + filePath;

        // No request-level timeout: files can be several GB and take many minutes.
        HttpResponse<InputStream> resp = http.send(
                authed(HttpRequest.newBuilder(URI.create(url)).GET()).build(),
                HttpResponse.BodyHandlers.ofInputStream());
        checkStatus(resp.statusCode(), "(binary response)");

        long total = resp.headers().firstValueAsLong("content-length").orElse(-1L);
        Path part  = dest.resolveSibling(dest.getFileName() + ".part");

        Files.createDirectories(dest.getParent());
        byte[] buf = new byte[64 * 1024]; // 64 KB per read
        long downloaded = 0;
        try (InputStream in = resp.body();
             var out = Files.newOutputStream(part)) {
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
                downloaded += n;
                if (onProgress != null) onProgress.accept(downloaded, total);
            }
        } catch (Exception e) {
            Files.deleteIfExists(part);
            throw e;
        }
        Files.move(part, dest, StandardCopyOption.REPLACE_EXISTING);
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    private HttpRequest.Builder authed(HttpRequest.Builder b) {
        return token != null ? b.header("Authorization", "Bearer " + token) : b;
    }

    private static void checkStatus(int code, String body) {
        if (code == 200) return;
        if (code == 401 || code == 403) {
            throw new RuntimeException(
                "Access denied (HTTP " + code + "). " +
                "For private or gated models, pass a HuggingFace token: " +
                "--token <token>  or  export HF_TOKEN=<token>");
        }
        if (code == 404) {
            throw new RuntimeException(
                "Not found (HTTP 404). Check that the owner, repo name, and file path are correct.");
        }
        String snippet = body.length() > 300 ? body.substring(0, 300) + "..." : body;
        throw new RuntimeException("HuggingFace request failed (HTTP " + code + "): " + snippet);
    }
}
