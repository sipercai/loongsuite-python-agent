# -*- coding: utf-8 -*-
"""Test cases for request attributes extractor."""

import json
import unittest
from typing import Any, Dict, List

from opentelemetry.instrumentation.agentscope.v1._request_attributes_extractor import (
    extract_unified_messages,
    _detect_message_format,
)


class TestRequestAttributesExtractor(unittest.TestCase):
    """Test cases for request attributes extractor."""

    def setUp(self) -> None:
        """Set up test cases."""
        # OpenAI格式示例
        self.openai_messages = [
            {
                "role": "system",
                "name": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You're a helpful assistant.",
                    }
                ],
            },
            {
                "role": "user",
                "name": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Weather in Paris?",
                    }
                ],
            },
            {
                "role": "assistant",
                "name": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                "content": "rainy, 57°F",
                "name": "get_weather",
            },
        ]

        # Anthropic格式示例
        self.anthropic_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You're a helpful assistant.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Weather in Paris?",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {
                            "location": "Paris",
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                        "content": [
                            {
                                "type": "text",
                                "text": "rainy, 57°F",
                            }
                        ],
                    }
                ],
            },
        ]

        # Gemini格式示例
        self.gemini_messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "You're a helpful assistant.",
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Weather in Paris?",
                    }
                ],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                            "name": "get_weather",
                            "args": {
                                "location": "Paris",
                            },
                        },
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                            "name": "get_weather",
                            "response": {
                                "output": "rainy, 57°F",
                            },
                        },
                    }
                ],
            },
        ]

        # 期望的统一格式
        self.expected_unified = [
            {
                "role": "system",
                "parts": [
                    {
                        "type": "text",
                        "content": "You're a helpful assistant.",
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "content": "Weather in Paris?",
                    }
                ],
            },
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                        "name": "get_weather",
                        "arguments": {
                            "location": "Paris",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call_VSPygqKTWdrhaFErNvMV18Yl",
                        "result": "rainy, 57°F",
                    }
                ],
            },
        ]

    def test_detect_openai_format(self) -> None:
        """Test OpenAI format detection."""
        detected = _detect_message_format(self.openai_messages)
        self.assertEqual(detected, "openai")

    def test_detect_anthropic_format(self) -> None:
        """Test Anthropic format detection."""
        detected = _detect_message_format(self.anthropic_messages)
        self.assertEqual(detected, "anthropic")

    def test_detect_gemini_format(self) -> None:
        """Test Gemini format detection."""
        detected = _detect_message_format(self.gemini_messages)
        self.assertEqual(detected, "gemini")

    def test_convert_openai_format(self) -> None:
        """Test OpenAI format conversion."""
        result = extract_unified_messages(self.openai_messages, provider="openai")
        
        # 验证基本结构
        self.assertEqual(len(result), 4)
        
        # 验证system消息
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["parts"][0]["type"], "text")
        self.assertEqual(result[0]["parts"][0]["content"], "You're a helpful assistant.")
        
        # 验证用户消息
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["parts"][0]["content"], "Weather in Paris?")
        
        # 验证assistant的tool call
        self.assertEqual(result[2]["role"], "assistant")
        self.assertEqual(result[2]["parts"][0]["type"], "tool_call")
        self.assertEqual(result[2]["parts"][0]["name"], "get_weather")
        self.assertEqual(result[2]["parts"][0]["arguments"]["location"], "Paris")
        
        # 验证tool response
        self.assertEqual(result[3]["role"], "tool")
        self.assertEqual(result[3]["parts"][0]["type"], "tool_call_response")
        self.assertEqual(result[3]["parts"][0]["result"], "rainy, 57°F")

    def test_convert_anthropic_format(self) -> None:
        """Test Anthropic format conversion."""
        result = extract_unified_messages(self.anthropic_messages, provider="anthropic")
        
        # 验证基本结构
        self.assertEqual(len(result), 4)
        
        # 验证system消息
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["parts"][0]["content"], "You're a helpful assistant.")
        
        # 验证tool call
        self.assertEqual(result[2]["role"], "assistant")
        self.assertEqual(result[2]["parts"][0]["type"], "tool_call")
        self.assertEqual(result[2]["parts"][0]["name"], "get_weather")

    def test_convert_gemini_format(self) -> None:
        """Test Gemini format conversion."""
        result = extract_unified_messages(self.gemini_messages, provider="gemini")
        
        # 验证基本结构
        self.assertEqual(len(result), 4)
        
        # 验证model角色转换为assistant
        assistant_msg = next((msg for msg in result if msg["role"] == "assistant"), None)
        self.assertIsNotNone(assistant_msg)
        self.assertEqual(assistant_msg["parts"][0]["type"], "tool_call")

    def test_auto_detection_and_conversion(self) -> None:
        """Test automatic format detection and conversion."""
        # 测试OpenAI自动检测
        result_openai = extract_unified_messages(self.openai_messages)
        self.assertEqual(len(result_openai), 4)
        
        # 测试Anthropic自动检测
        result_anthropic = extract_unified_messages(self.anthropic_messages)
        self.assertEqual(len(result_anthropic), 4)
        
        # 测试Gemini自动检测
        result_gemini = extract_unified_messages(self.gemini_messages)
        self.assertEqual(len(result_gemini), 4)

    def test_empty_messages(self) -> None:
        """Test handling of empty message list."""
        result = extract_unified_messages([])
        self.assertEqual(result, [])

    def test_image_handling(self) -> None:
        """Test image content handling."""
        openai_with_image = [
            {
                "role": "user",
                "name": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANS..."
                        },
                    },
                ],
            }
        ]
        
        result = extract_unified_messages(openai_with_image, provider="openai")
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["parts"]), 2)
        self.assertEqual(result[0]["parts"][0]["type"], "text")
        self.assertEqual(result[0]["parts"][1]["type"], "image")

    def test_default_format_handling(self) -> None:
        """Test default format handling for unknown formats."""
        unknown_format = [
            {
                "role": "user",
                "message": "Hello world",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ]
        
        result = extract_unified_messages(unknown_format)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        # 应该将unknown内容转换为文本
        self.assertTrue(len(result[0]["parts"]) > 0)


if __name__ == "__main__":
    unittest.main()