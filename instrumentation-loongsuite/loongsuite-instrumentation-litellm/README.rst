LoongSuite LiteLLM Instrumentation
======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-litellm.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-litellm/

This library provides automatic instrumentation for the
`LiteLLM <https://github.com/BerriAI/litellm>`_ library, which provides
a unified interface to 100+ LLM providers.

Installation
------------

::

     git clone https://github.com/alibaba/loongsuite-python-agent.git
     cd loongsuite-python-agent
     pip install ./instrumentation-loongsuite/loongsuite-instrumentation-litellm

Configuration
-------------

The instrumentation can be enabled/disabled using environment variables:

* ``ENABLE_LITELLM_INSTRUMENTOR``: Enable/disable instrumentation (default: true)
* ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``: Enable GenAI semantic conventions
* ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``: Set to ``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
    import litellm

    # Instrument LiteLLM
    LiteLLMInstrumentor().instrument()

    # Use LiteLLM as normal
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Local OTLP smoke
----------------

The ``examples/litellm_genai_smoke.py`` script sends real LiteLLM traffic for:

* non-streaming completion
* streaming completion
* concurrent async completion calls

Set ``LITELLM_SMOKE_MODE`` to ``non_streaming``, ``streaming``,
``concurrent``, or ``all`` (default) to run a subset.

Example with a local ``otel-gui`` OTLP endpoint:

.. code:: console

    export DASHSCOPE_API_KEY=...
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
    export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
    export OTEL_SERVICE_NAME=loongsuite-litellm-smoke

    loongsuite-instrument python \
        instrumentation-loongsuite/loongsuite-instrumentation-litellm/examples/litellm_genai_smoke.py

Features
--------

This instrumentation automatically captures:

* LLM completion calls (sync and async)
* Streaming completions
* Embedding calls
* Retry mechanisms
* Tool/function calls
* Request and response metadata
* Token usage
* Model information

The instrumentation follows OpenTelemetry semantic conventions for GenAI operations.

References
----------

* `OpenTelemetry LiteLLM Instrumentation <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/litellm/litellm.html>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `LiteLLM Documentation <https://docs.litellm.ai/>`_
