package dev.localllm.model;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utilities for detecting and enumerating split GGUF models.
 *
 * <p>Three naming conventions are recognized:
 * <ol>
 *   <li>{@code name-NNNNN-of-MMMMM.gguf}  — llama.cpp standard (most common)</li>
 *   <li>{@code name.gguf-split-a}, {@code name.gguf-split-b}, …  — older format</li>
 *   <li>{@code name.gguf.part0}, {@code name.gguf.part1}, …  — alternative format</li>
 * </ol>
 *
 * <p>For pattern 1, the total shard count is embedded in the filename, so all expected
 * paths can be generated without touching the filesystem (useful for remote operations).
 * For patterns 2 and 3, shards are discovered by probing the filesystem.
 */
public final class SplitGguf {

    /** -NNNNN-of-MMMMM.gguf (llama.cpp standard) */
    public static final Pattern PATTERN1 = Pattern.compile(
            "^(.+?)-(\\d{5})-of-(\\d{5})\\.gguf$", Pattern.CASE_INSENSITIVE);

    /** .gguf-split-a, .gguf-split-b, … */
    private static final Pattern PATTERN2 = Pattern.compile(
            "^(.+)\\.gguf-split-([a-z])$", Pattern.CASE_INSENSITIVE);

    /** .gguf.part0, .gguf.part1, … */
    private static final Pattern PATTERN3 = Pattern.compile(
            "^(.+)\\.gguf\\.part(\\d+)$", Pattern.CASE_INSENSITIVE);

    private SplitGguf() {}

    /** Describes a detected set of related GGUF shards. */
    public static final class Split {
        /** All shard paths in order. For pattern 1 this includes paths that may not yet exist. */
        public final List<Path> shards;
        /** 1-based index of the file that was passed to {@link #detect}. */
        public final int shardIndex;
        /** Total number of shards. */
        public final int totalShards;

        Split(List<Path> shards, int index, int total) {
            this.shards      = Collections.unmodifiableList(shards);
            this.shardIndex  = index;
            this.totalShards = total;
        }

        /** First (primary) shard — the path to pass to llama.cpp. */
        public Path first() { return shards.get(0); }

        /** True if the detected file is the primary shard (index 1). */
        public boolean isFirstShard() { return shardIndex == 1; }

        /** Returns only the shard paths that currently exist on disk. */
        public List<Path> existingShards() {
            List<Path> out = new ArrayList<>();
            for (Path p : shards) if (Files.exists(p)) out.add(p);
            return out;
        }
    }

    /**
     * Detect whether {@code path} is part of a split GGUF.
     * Returns {@code null} for standalone GGUFs.
     *
     * <p>The returned {@link Split} always lists shards starting from index 1,
     * regardless of which shard was passed in.
     */
    public static Split detect(Path path) {
        String name = path.getFileName().toString();
        Path   dir  = path.getParent() != null ? path.getParent() : Paths.get(".");

        // Pattern 1: name-NNNNN-of-MMMMM.gguf
        Matcher m1 = PATTERN1.matcher(name);
        if (m1.matches()) {
            String base  = m1.group(1);
            int    index = Integer.parseInt(m1.group(2));
            int    total = Integer.parseInt(m1.group(3));
            List<Path> shards = new ArrayList<>(total);
            for (int i = 1; i <= total; i++) {
                shards.add(dir.resolve(String.format("%s-%05d-of-%05d.gguf", base, i, total)));
            }
            return new Split(shards, index, total);
        }

        // Pattern 2: name.gguf-split-a,b,… (probe until missing)
        Matcher m2 = PATTERN2.matcher(name);
        if (m2.matches()) {
            String base   = m2.group(1);
            char   letter = Character.toLowerCase(m2.group(2).charAt(0));
            int    index  = letter - 'a' + 1;
            List<Path> shards = new ArrayList<>();
            for (char c = 'a'; c <= 'z'; c++) {
                Path shard = dir.resolve(base + ".gguf-split-" + c);
                if (!Files.exists(shard)) break;
                shards.add(shard);
            }
            if (shards.isEmpty()) shards.add(path);
            return new Split(shards, index, shards.size());
        }

        // Pattern 3: name.gguf.part0,1,… (probe until missing)
        Matcher m3 = PATTERN3.matcher(name);
        if (m3.matches()) {
            String base  = m3.group(1);
            int    part  = Integer.parseInt(m3.group(2));
            int    index = part + 1; // 0-based → 1-based
            List<Path> shards = new ArrayList<>();
            for (int i = 0; i <= 9999; i++) {
                Path shard = dir.resolve(base + ".gguf.part" + i);
                if (!Files.exists(shard)) break;
                shards.add(shard);
            }
            if (shards.isEmpty()) shards.add(path);
            return new Split(shards, index, shards.size());
        }

        return null; // standalone GGUF
    }

    /**
     * Parse shard metadata from a bare filename (no filesystem access).
     * Returns {@code [shardIndex, totalShards]} (both 1-based) for pattern-1 filenames,
     * or {@code null} if the filename does not match a known split pattern.
     *
     * <p>Useful for classifying remote filenames before they are downloaded.
     */
    public static int[] parseShardInfo(String filename) {
        // Strip any directory prefix
        int slash = filename.lastIndexOf('/');
        String file = slash >= 0 ? filename.substring(slash + 1) : filename;
        Matcher m = PATTERN1.matcher(file);
        if (m.matches()) {
            return new int[]{Integer.parseInt(m.group(2)), Integer.parseInt(m.group(3))};
        }
        return null;
    }

    /**
     * Generate remote file paths for all shards of a split model, given the remote path
     * of any one shard (pattern 1 only).
     *
     * <p>Example: {@code remoteShardPaths("blobs/model-00001-of-00004.gguf", 4)} returns
     * {@code ["blobs/model-00001-of-00004.gguf", ..., "blobs/model-00004-of-00004.gguf"]}.
     *
     * @param anyShardRemotePath remote file path of any shard
     * @param total              total shard count (from the filename)
     * @return list of all remote shard paths; or a singleton list if pattern is unrecognized
     */
    public static List<String> remoteShardPaths(String anyShardRemotePath, int total) {
        int slash = anyShardRemotePath.lastIndexOf('/');
        String dir  = slash >= 0 ? anyShardRemotePath.substring(0, slash + 1) : "";
        String file = slash >= 0 ? anyShardRemotePath.substring(slash + 1) : anyShardRemotePath;
        Matcher m = PATTERN1.matcher(file);
        if (!m.matches()) return Collections.singletonList(anyShardRemotePath);
        String base = m.group(1);
        List<String> result = new ArrayList<>(total);
        for (int i = 1; i <= total; i++) {
            result.add(dir + String.format("%s-%05d-of-%05d.gguf", base, i, total));
        }
        return result;
    }
}
