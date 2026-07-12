package dev.localllm.model;

import java.util.List;

/**
 * Persistent metadata for one registered model. Fields from a Modelfile
 * ({@code temperature}, {@code numPredict}, {@code numCtx}, {@code numThreads},
 * {@code systemPrompt}) are nullable: a {@code null} value means "use the
 * runtime default" rather than "explicitly not set to anything".
 */
public class ModelConfig {
    private String name;
    private String path;
    private String binary;
    private String format;
    private long sizeBytes;
    private String addedAt;

    // Split GGUF support: null for single-file models.
    // For multi-shard models, contains all shard paths in order (including path == shards.get(0)).
    // sizeBytes is the sum of all shard sizes.
    private List<String> shards;

    // Modelfile parameters (null = not configured, fall back to runtime default)
    private Float   temperature;
    private Integer numPredict;
    private Integer numCtx;
    private Integer numThreads;
    private String  systemPrompt;

    // SHA-256 hex digest of the GGUF file (null = not yet computed)
    private String  sha256;

    // GGUF header metadata (populated by GgufReader at add/create/pull time; null if unavailable)
    private String  ggufArchitecture;
    private String  ggufQuantization;
    private Long    ggufParameterCount;
    private Integer ggufContextLength;
    private Integer ggufBlockCount;
    private Integer ggufEmbeddingLength;

    public String getName()  { return name; }
    public void setName(String name) { this.name = name; }

    public String getPath()  { return path; }
    public void setPath(String path) { this.path = path; }

    public String getBinary() { return binary; }
    public void setBinary(String binary) { this.binary = binary; }

    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }

    public long getSizeBytes() { return sizeBytes; }
    public void setSizeBytes(long sizeBytes) { this.sizeBytes = sizeBytes; }

    public String getAddedAt() { return addedAt; }
    public void setAddedAt(String addedAt) { this.addedAt = addedAt; }

    public List<String> getShards()             { return shards; }
    public void         setShards(List<String> shards) { this.shards = shards; }
    public boolean      isSplit()               { return shards != null && shards.size() > 1; }

    public Float   getTemperature() { return temperature; }
    public void setTemperature(Float temperature) { this.temperature = temperature; }

    public Integer getNumPredict() { return numPredict; }
    public void setNumPredict(Integer numPredict) { this.numPredict = numPredict; }

    public Integer getNumCtx() { return numCtx; }
    public void setNumCtx(Integer numCtx) { this.numCtx = numCtx; }

    public Integer getNumThreads() { return numThreads; }
    public void setNumThreads(Integer numThreads) { this.numThreads = numThreads; }

    public String getSystemPrompt() { return systemPrompt; }
    public void setSystemPrompt(String systemPrompt) { this.systemPrompt = systemPrompt; }

    public String  getSha256()             { return sha256; }
    public void    setSha256(String v)     { this.sha256 = v; }

    public String  getGgufArchitecture()   { return ggufArchitecture; }
    public void    setGgufArchitecture(String v)   { this.ggufArchitecture   = v; }

    public String  getGgufQuantization()   { return ggufQuantization; }
    public void    setGgufQuantization(String v)   { this.ggufQuantization   = v; }

    public Long    getGgufParameterCount() { return ggufParameterCount; }
    public void    setGgufParameterCount(Long v)   { this.ggufParameterCount = v; }

    public Integer getGgufContextLength()  { return ggufContextLength; }
    public void    setGgufContextLength(Integer v) { this.ggufContextLength  = v; }

    public Integer getGgufBlockCount()     { return ggufBlockCount; }
    public void    setGgufBlockCount(Integer v)    { this.ggufBlockCount     = v; }

    public Integer getGgufEmbeddingLength(){ return ggufEmbeddingLength; }
    public void    setGgufEmbeddingLength(Integer v){ this.ggufEmbeddingLength = v; }
}
