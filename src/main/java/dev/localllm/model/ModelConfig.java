package dev.localllm.model;

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

    // Modelfile parameters (null = not configured, fall back to runtime default)
    private Float   temperature;
    private Integer numPredict;
    private Integer numCtx;
    private Integer numThreads;
    private String  systemPrompt;

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
}
