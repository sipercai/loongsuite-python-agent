# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial Hermes Agent instrumentation package
  - Agent, step, LLM, and tool spans built on `opentelemetry-util-genai`
  - Live and spec test coverage for Hermes agent conversations
  - Provider normalization for Hermes agent spans and underlying LLM spans
