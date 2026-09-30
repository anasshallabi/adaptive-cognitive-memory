# GitHub repository and local development

Canonical public repository: https://github.com/anasshallabi/adaptive-cognitive-memory

## Clone

```bash
git clone https://github.com/anasshallabi/adaptive-cognitive-memory.git
cd adaptive-cognitive-memory
python -m unittest discover -s tests -v
python -m examples.one_shot
```

## Contribute

Create a feature branch and submit a pull request. Include tests, a clear hypothesis, evaluation methodology, dataset licensing/provenance, and known limitations. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Privacy and reproducibility

Never commit API tokens, `.env` files, private images, credentials, or third-party datasets without redistribution permission. Review `git status` and `git diff --cached` before pushing. For privacy, set Git's author email to a GitHub `noreply` address.
