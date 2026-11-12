# Third-party notices

## markdown-memory (MIT)

`src/memidx/embed.py` contains code derived from `markdown-memory` 0.5.0
(https://github.com/hishamkaram/markdown-memory, author Hesham Karm): the EmbeddingGemma model
constants (repository, pinned revision, graph file, file manifest with sizes and checksums), the query and
document prompt prefixes, the tokenizer settings, the batching, the use of the `sentence_embedding` output with
L2 normalization, and the onnxruntime session options. memidx does not depend on that package.

```
MIT License

Copyright (c) 2026 heshamkarm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## EmbeddingGemma-300m (not included)

The embedding model is not part of memidx and is never bundled. `memidx model download` fetches the ONNX export
`onnx-community/embeddinggemma-300m-ONNX` (pinned revision) from Hugging Face. The model is distributed by Google
under the Gemma Terms of Use (https://ai.google.dev/gemma/terms). You must read and accept those terms yourself
before you download or use the model.

## Runtime dependencies

Optional extras install third-party packages under their own licenses: `numpy`, `onnxruntime`, `tokenizers`,
`huggingface_hub` (extra `vectors`), `mcp` (extra `mcp`), `tzdata` (Windows only).
