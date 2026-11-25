package io.teknek.reckoning.generator2;

/** Explicit per-forward state passed through model layers. */
public interface ForwardContext {
    ForwardContext NONE = new ForwardContext() { };
}
