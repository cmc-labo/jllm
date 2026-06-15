package dev.localllm.model;

public class ModelConfig {
    private String name;
    private String path;
    private String binary;
    private String format;
    private long sizeBytes;
    private String addedAt;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getPath() { return path; }
    public void setPath(String path) { this.path = path; }

    public String getBinary() { return binary; }
    public void setBinary(String binary) { this.binary = binary; }

    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }

    public long getSizeBytes() { return sizeBytes; }
    public void setSizeBytes(long sizeBytes) { this.sizeBytes = sizeBytes; }

    public String getAddedAt() { return addedAt; }
    public void setAddedAt(String addedAt) { this.addedAt = addedAt; }
}
