# ADR-001: PyTorch over TensorFlow for Model Implementation

**Date:** 2025-10-25
**Status:** Accepted
**Deciders:** System Architecture Team
**Technical Story:** Model Implementation Framework Selection

---

## Context and Problem Statement

We need to select a deep learning framework for implementing the Bayesian Expectation Transformer. The choice will impact development velocity, model flexibility, deployment options, and long-term maintainability.

**Decision Drivers:**
- Research-oriented development (rapid prototyping)
- Custom layer implementation complexity
- Community support and ecosystem
- Production deployment requirements
- Performance characteristics
- Integration with existing tools (HuggingFace)

---

## Considered Options

### Option 1: PyTorch
**Pros:**
- Pythonic, imperative programming style (easier debugging)
- Dynamic computational graphs (better for research)
- Strong academic adoption (easier to implement paper algorithms)
- Excellent HuggingFace Transformers integration
- Better support for custom autograd functions
- Growing production ecosystem (TorchServe, TorchScript)

**Cons:**
- Historically weaker production tooling vs TensorFlow
- Less mature mobile deployment (though improving)
- Smaller ecosystem of pre-trained models (though HuggingFace bridges this)

### Option 2: TensorFlow/Keras
**Pros:**
- More mature production deployment (TF Serving)
- Better mobile/edge support (TensorFlow Lite)
- Comprehensive production ecosystem
- TensorBoard integration

**Cons:**
- Eager execution still feels less natural than PyTorch
- More verbose custom layer implementation
- Static graph paradigm less flexible for research
- HuggingFace support exists but PyTorch is primary

### Option 3: JAX
**Pros:**
- Functional programming paradigm
- Excellent performance (XLA compilation)
- Composable transformations (jit, grad, vmap)
- Growing research adoption

**Cons:**
- Steeper learning curve (functional paradigm)
- Smaller ecosystem and community
- Less mature production tooling
- Limited HuggingFace integration

---

## Decision Outcome

**Chosen option: PyTorch**

### Rationale

1. **Research-First Approach:** Our implementation requires custom components (MartingaleAwareAttention, OptimalCoTLayer) that are easier to develop and debug in PyTorch's imperative style.

2. **HuggingFace Ecosystem:** Seamless integration with HuggingFace Transformers and Datasets libraries is critical for:
   - Pre-trained tokenizers
   - Dataset loading (IMDB)
   - Future model sharing on HuggingFace Hub

3. **Custom Autograd:** We need custom backward passes for:
   - Permutation averaging (variance reduction)
   - MDL regularization
   - Sufficient statistics computation

   PyTorch makes custom autograd straightforward with `torch.autograd.Function`.

4. **Dynamic Graphs:** The Bayesian components require runtime decisions:
   - Optimal CoT length computation (varies per input)
   - Adaptive permutation counts (based on sequence length)
   - Conditional uncertainty estimation

   Dynamic graphs make these patterns natural.

5. **Community and Resources:** PyTorch is the de facto standard in research:
   - More papers implement in PyTorch
   - Better community support for novel architectures
   - Easier to recruit developers familiar with PyTorch

### Positive Consequences

- Faster development velocity for novel components
- Easier debugging with imperative code
- Natural integration with HuggingFace ecosystem
- Strong community support for research-oriented features
- Clear path to production (TorchServe, TorchScript, ONNX)

### Negative Consequences

- Need to learn TorchServe for production deployment (vs TensorFlow Serving)
- Mobile deployment requires additional work (TorchScript Mobile)
- Slightly higher memory usage than TensorFlow in some cases

### Mitigation Strategies

**Production Deployment:**
- Use TorchScript for model serialization
- Export to ONNX for maximum deployment flexibility
- Containerize with Docker for consistent environments

**Mobile Deployment (if needed):**
- Use TorchScript Mobile
- Consider ONNX Runtime for cross-platform support
- Quantization for model size reduction

---

## Validation

### Success Criteria

✅ **Met:**
- Custom layers implemented in <2 weeks
- Model training works on IMDB dataset
- HuggingFace integration seamless
- Code is maintainable and well-documented

🔄 **In Progress:**
- Production deployment strategy
- ONNX export for inference optimization

⏳ **Planned:**
- Mobile deployment proof-of-concept

---

## Related Decisions

- [ADR-002: FastAPI for REST API](ADR-002-fastapi-for-rest-api.md) - API framework selection
- [ADR-003: HuggingFace Datasets Integration](ADR-003-huggingface-datasets.md) - Data pipeline

---

## References

1. PyTorch Documentation: https://pytorch.org/docs/
2. TorchServe: https://pytorch.org/serve/
3. HuggingFace Transformers: https://huggingface.co/docs/transformers/
4. ONNX: https://onnx.ai/

---

**Last Updated:** 2025-10-25
**Next Review:** 2026-01-25
