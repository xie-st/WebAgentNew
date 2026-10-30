"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useState } from "react";
import { FiAward, FiCpu, FiZap, FiChevronDown, FiArrowUp, FiClock, FiTrendingUp, FiBarChart2 } from "react-icons/fi";
import type { IconType } from "react-icons";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { usePyPIVersion } from "@/components/PyPIVersion";
import { accentTextStyle } from "@/components/accentText";

/* ── Benchmark metadata ── */

const benchmarkCategories = [
  { name: "Calculator", color: "#ffd700", benchmarks: ["GSM8K", "GSM-PLUS", "MATH-500"] },
  { name: "Math reasoning", color: "#00ff88", benchmarks: ["BB-Easy", "BB-Med", "BB-Hard"] },
  { name: "Agentic", color: "#ff6b6b", benchmarks: ["GAIA", "SimpleQA"] },
  { name: "Memory", color: "#a78bfa", benchmarks: ["LoCoMo", "LongMemEval"] },
  { name: "Retrieval", color: "#00e5ff", benchmarks: ["ARC-C", "ARC-E", "CSQA"] },
];

const allBenchmarks = ["GSM8K", "GSM-PLUS", "MATH-500", "BB-Easy", "BB-Med", "BB-Hard", "GAIA", "SimpleQA", "LoCoMo", "LongMemEval", "ARC-C", "ARC-E", "CSQA"];

function getBenchmarkCategory(b: string) {
  for (const cat of benchmarkCategories) {
    if (cat.benchmarks.includes(b)) return cat;
  }
  return benchmarkCategories[0];
}

/* ── Model data from paper ── */

interface ModelData {
  rank: number;
  name: string;
  family: string;
  size: string;
  sizeNum: number;
  effgenAvg: number;
  effgenScores: Record<string, number>;
  frameworkAvgs: { raw: number; langchain: number; autogen: number; smolagents: number };
  frameworkScores: Record<string, Record<string, number>>;
  improvement: number;
  timing?: { effgen: number; raw: number; langchain: number; autogen: number; smolagents: number };
}

const models: ModelData[] = [
  {
    rank: 1, name: "Qwen2.5-32B-Instruct", family: "Qwen", size: "32B", sizeNum: 32,
    effgenAvg: 70.97,
    effgenScores: { "GSM8K": 95.75, "GSM-PLUS": 78.38, "MATH-500": 75.40, "BB-Easy": 96.67, "BB-Med": 64.20, "BB-Hard": 58.86, "GAIA": 28.12, "SimpleQA": 84.00, "LoCoMo": 31.04, "LongMemEval": 35.64, "ARC-C": 92.50, "ARC-E": 95.29, "CSQA": 86.80 },
    frameworkAvgs: { raw: 58.32, langchain: 61.03, autogen: 64.99, smolagents: 63.85 },
    frameworkScores: {
      raw: { "GSM8K": 95.60, "GSM-PLUS": 77.42, "MATH-500": 73.00, "BB-Easy": 71.56, "BB-Med": 48.40, "BB-Hard": 26.53, "GAIA": 15.62, "SimpleQA": 14.00, "LoCoMo": 31.91, "LongMemEval": 30.12, "ARC-C": 92.92, "ARC-E": 94.32, "CSQA": 86.81 },
      langchain: { "GSM8K": 95.07, "GSM-PLUS": 75.83, "MATH-500": 59.20, "BB-Easy": 88.05, "BB-Med": 60.00, "BB-Hard": 27.14, "GAIA": 18.75, "SimpleQA": 54.00, "LoCoMo": 25.07, "LongMemEval": 18.66, "ARC-C": 93.34, "ARC-E": 93.94, "CSQA": 84.28 },
      autogen: { "GSM8K": 94.54, "GSM-PLUS": 76.96, "MATH-500": 72.80, "BB-Easy": 92.44, "BB-Med": 63.60, "BB-Hard": 30.67, "GAIA": 21.87, "SimpleQA": 70.00, "LoCoMo": 30.37, "LongMemEval": 28.22, "ARC-C": 89.25, "ARC-E": 92.51, "CSQA": 81.65 },
      smolagents: { "GSM8K": 93.40, "GSM-PLUS": 74.83, "MATH-500": 68.00, "BB-Easy": 94.63, "BB-Med": 68.80, "BB-Hard": 45.83, "GAIA": 15.62, "SimpleQA": 52.00, "LoCoMo": 17.07, "LongMemEval": 27.22, "ARC-C": 93.26, "ARC-E": 94.44, "CSQA": 84.93 },
    },
    improvement: 5.98,
    timing: { effgen: 48.0, raw: 180.1, langchain: 385.9, autogen: 212.7, smolagents: 265.6 },
  },
  {
    rank: 2, name: "Gemma3-27B", family: "Google", size: "27B", sizeNum: 27,
    effgenAvg: 69.19,
    effgenScores: { "GSM8K": 94.11, "GSM-PLUS": 76.54, "MATH-500": 73.60, "BB-Easy": 95.12, "BB-Med": 62.20, "BB-Hard": 56.67, "GAIA": 25.00, "SimpleQA": 80.00, "LoCoMo": 30.18, "LongMemEval": 34.72, "ARC-C": 91.64, "ARC-E": 94.25, "CSQA": 85.42 },
    frameworkAvgs: { raw: 56.09, langchain: 59.36, autogen: 63.31, smolagents: 62.31 },
    frameworkScores: {
      raw: { "GSM8K": 93.25, "GSM-PLUS": 74.58, "MATH-500": 71.20, "BB-Easy": 69.76, "BB-Med": 46.40, "BB-Hard": 24.58, "GAIA": 15.62, "SimpleQA": 12.00, "LoCoMo": 25.84, "LongMemEval": 29.45, "ARC-C": 91.38, "ARC-E": 93.52, "CSQA": 81.65 },
      langchain: { "GSM8K": 93.08, "GSM-PLUS": 73.25, "MATH-500": 57.80, "BB-Easy": 86.34, "BB-Med": 58.00, "BB-Hard": 26.43, "GAIA": 18.75, "SimpleQA": 50.00, "LoCoMo": 24.12, "LongMemEval": 17.42, "ARC-C": 92.56, "ARC-E": 93.68, "CSQA": 80.24 },
      autogen: { "GSM8K": 92.67, "GSM-PLUS": 74.12, "MATH-500": 70.80, "BB-Easy": 90.73, "BB-Med": 61.60, "BB-Hard": 28.75, "GAIA": 21.87, "SimpleQA": 66.00, "LoCoMo": 29.45, "LongMemEval": 27.18, "ARC-C": 88.14, "ARC-E": 91.35, "CSQA": 80.42 },
      smolagents: { "GSM8K": 91.82, "GSM-PLUS": 72.46, "MATH-500": 66.40, "BB-Easy": 92.93, "BB-Med": 66.80, "BB-Hard": 43.75, "GAIA": 15.62, "SimpleQA": 48.00, "LoCoMo": 16.24, "LongMemEval": 26.35, "ARC-C": 92.18, "ARC-E": 93.94, "CSQA": 83.58 },
    },
    improvement: 5.88,
  },
  {
    rank: 3, name: "GPT-OSS-20B", family: "OpenAI", size: "20B", sizeNum: 20,
    effgenAvg: 67.82,
    effgenScores: { "GSM8K": 90.28, "GSM-PLUS": 70.15, "MATH-500": 64.60, "BB-Easy": 94.88, "BB-Med": 76.80, "BB-Hard": 63.33, "GAIA": 21.87, "SimpleQA": 74.00, "LoCoMo": 25.47, "LongMemEval": 34.28, "ARC-C": 88.52, "ARC-E": 93.14, "CSQA": 84.35 },
    frameworkAvgs: { raw: 56.75, langchain: 55.93, autogen: 57.68, smolagents: 53.37 },
    frameworkScores: {
      raw: { "GSM8K": 88.75, "GSM-PLUS": 68.42, "MATH-500": 62.80, "BB-Easy": 84.17, "BB-Med": 61.91, "BB-Hard": 51.57, "GAIA": 12.50, "SimpleQA": 8.00, "LoCoMo": 21.36, "LongMemEval": 28.92, "ARC-C": 81.45, "ARC-E": 89.27, "CSQA": 78.63 },
      langchain: { "GSM8K": 85.32, "GSM-PLUS": 64.75, "MATH-500": 50.40, "BB-Easy": 86.59, "BB-Med": 68.40, "BB-Hard": 48.33, "GAIA": 9.38, "SimpleQA": 38.00, "LoCoMo": 10.25, "LongMemEval": 16.58, "ARC-C": 83.72, "ARC-E": 88.94, "CSQA": 77.45 },
      autogen: { "GSM8K": 88.14, "GSM-PLUS": 67.58, "MATH-500": 61.60, "BB-Easy": 89.27, "BB-Med": 65.20, "BB-Hard": 52.14, "GAIA": 12.50, "SimpleQA": 50.00, "LoCoMo": 17.42, "LongMemEval": 26.73, "ARC-C": 77.38, "ARC-E": 81.65, "CSQA": 60.28 },
      smolagents: { "GSM8K": 85.68, "GSM-PLUS": 63.92, "MATH-500": 18.40, "BB-Easy": 92.44, "BB-Med": 71.20, "BB-Hard": 46.67, "GAIA": 9.38, "SimpleQA": 34.00, "LoCoMo": 11.48, "LongMemEval": 27.24, "ARC-C": 73.15, "ARC-E": 82.38, "CSQA": 77.92 },
    },
    improvement: 10.14,
  },
  {
    rank: 4, name: "Qwen2.5-14B-Instruct", family: "Qwen", size: "14B", sizeNum: 14,
    effgenAvg: 66.38,
    effgenScores: { "GSM8K": 94.84, "GSM-PLUS": 76.62, "MATH-500": 69.60, "BB-Easy": 92.27, "BB-Med": 57.60, "BB-Hard": 46.37, "GAIA": 18.75, "SimpleQA": 72.00, "LoCoMo": 27.94, "LongMemEval": 36.64, "ARC-C": 89.86, "ARC-E": 94.39, "CSQA": 86.00 },
    frameworkAvgs: { raw: 54.46, langchain: 55.95, autogen: 59.07, smolagents: 54.69 },
    frameworkScores: {
      raw: { "GSM8K": 93.78, "GSM-PLUS": 75.17, "MATH-500": 67.80, "BB-Easy": 64.63, "BB-Med": 38.80, "BB-Hard": 24.17, "GAIA": 12.50, "SimpleQA": 8.00, "LoCoMo": 26.92, "LongMemEval": 30.94, "ARC-C": 88.99, "ARC-E": 92.97, "CSQA": 83.37 },
      langchain: { "GSM8K": 89.99, "GSM-PLUS": 70.92, "MATH-500": 53.20, "BB-Easy": 74.71, "BB-Med": 47.20, "BB-Hard": 29.58, "GAIA": 12.50, "SimpleQA": 46.00, "LoCoMo": 19.00, "LongMemEval": 17.68, "ARC-C": 91.13, "ARC-E": 93.90, "CSQA": 81.49 },
      autogen: { "GSM8K": 94.09, "GSM-PLUS": 74.83, "MATH-500": 66.00, "BB-Easy": 77.07, "BB-Med": 51.20, "BB-Hard": 24.29, "GAIA": 15.62, "SimpleQA": 58.00, "LoCoMo": 17.15, "LongMemEval": 27.72, "ARC-C": 89.42, "ARC-E": 92.89, "CSQA": 79.69 },
      smolagents: { "GSM8K": 90.45, "GSM-PLUS": 71.17, "MATH-500": 20.43, "BB-Easy": 86.27, "BB-Med": 48.80, "BB-Hard": 34.29, "GAIA": 9.38, "SimpleQA": 46.00, "LoCoMo": 13.04, "LongMemEval": 24.34, "ARC-C": 90.87, "ARC-E": 93.14, "CSQA": 82.80 },
    },
    improvement: 7.31,
    timing: { effgen: 30.5, raw: 143.6, langchain: 282.0, autogen: 163.9, smolagents: 187.7 },
  },
  {
    rank: 5, name: "Qwen2.5-7B-Instruct", family: "Qwen", size: "7B", sizeNum: 7,
    effgenAvg: 63.07,
    effgenScores: { "GSM8K": 91.28, "GSM-PLUS": 68.12, "MATH-500": 66.80, "BB-Easy": 89.02, "BB-Med": 54.80, "BB-Hard": 28.57, "GAIA": 21.87, "SimpleQA": 76.00, "LoCoMo": 25.32, "LongMemEval": 35.34, "ARC-C": 87.88, "ARC-E": 91.92, "CSQA": 83.05 },
    frameworkAvgs: { raw: 50.77, langchain: 49.99, autogen: 51.72, smolagents: 48.19 },
    frameworkScores: {
      raw: { "GSM8K": 90.75, "GSM-PLUS": 72.54, "MATH-500": 65.80, "BB-Easy": 59.68, "BB-Med": 27.60, "BB-Hard": 15.42, "GAIA": 12.50, "SimpleQA": 6.00, "LoCoMo": 22.35, "LongMemEval": 29.88, "ARC-C": 83.79, "ARC-E": 90.91, "CSQA": 82.80 },
      langchain: { "GSM8K": 84.38, "GSM-PLUS": 67.79, "MATH-500": 41.20, "BB-Easy": 72.20, "BB-Med": 45.60, "BB-Hard": 12.86, "GAIA": 9.38, "SimpleQA": 34.00, "LoCoMo": 9.02, "LongMemEval": 16.32, "ARC-C": 86.95, "ARC-E": 90.99, "CSQA": 79.12 },
      autogen: { "GSM8K": 90.30, "GSM-PLUS": 71.25, "MATH-500": 64.60, "BB-Easy": 77.80, "BB-Med": 43.20, "BB-Hard": 7.14, "GAIA": 12.50, "SimpleQA": 46.00, "LoCoMo": 16.44, "LongMemEval": 27.83, "ARC-C": 76.28, "ARC-E": 80.39, "CSQA": 58.64 },
      smolagents: { "GSM8K": 84.00, "GSM-PLUS": 64.58, "MATH-500": 16.20, "BB-Easy": 80.24, "BB-Med": 49.20, "BB-Hard": 24.17, "GAIA": 9.38, "SimpleQA": 32.00, "LoCoMo": 10.25, "LongMemEval": 27.47, "ARC-C": 71.25, "ARC-E": 81.36, "CSQA": 76.33 },
    },
    improvement: 11.35,
    timing: { effgen: 17.7, raw: 58.1, langchain: 92.9, autogen: 52.9, smolagents: 136.7 },
  },
  {
    rank: 6, name: "Gemma3-12B", family: "Google", size: "12B", sizeNum: 12,
    effgenAvg: 60.92,
    effgenScores: { "GSM8K": 89.54, "GSM-PLUS": 66.28, "MATH-500": 64.80, "BB-Easy": 87.07, "BB-Med": 52.80, "BB-Hard": 26.90, "GAIA": 18.75, "SimpleQA": 70.00, "LoCoMo": 23.64, "LongMemEval": 33.82, "ARC-C": 86.35, "ARC-E": 90.68, "CSQA": 81.27 },
    frameworkAvgs: { raw: 49.10, langchain: 48.47, autogen: 49.98, smolagents: 46.65 },
    frameworkScores: {
      raw: { "GSM8K": 88.42, "GSM-PLUS": 69.83, "MATH-500": 63.40, "BB-Easy": 57.32, "BB-Med": 26.00, "BB-Hard": 14.17, "GAIA": 12.50, "SimpleQA": 6.00, "LoCoMo": 20.74, "LongMemEval": 28.65, "ARC-C": 82.13, "ARC-E": 89.72, "CSQA": 79.45 },
      langchain: { "GSM8K": 82.65, "GSM-PLUS": 65.42, "MATH-500": 40.20, "BB-Easy": 70.49, "BB-Med": 43.60, "BB-Hard": 12.14, "GAIA": 9.38, "SimpleQA": 30.00, "LoCoMo": 8.47, "LongMemEval": 15.23, "ARC-C": 85.27, "ARC-E": 89.94, "CSQA": 77.28 },
      autogen: { "GSM8K": 88.14, "GSM-PLUS": 68.92, "MATH-500": 62.40, "BB-Easy": 75.61, "BB-Med": 41.20, "BB-Hard": 6.67, "GAIA": 12.50, "SimpleQA": 42.00, "LoCoMo": 15.38, "LongMemEval": 26.47, "ARC-C": 74.63, "ARC-E": 78.85, "CSQA": 56.92 },
      smolagents: { "GSM8K": 82.41, "GSM-PLUS": 62.73, "MATH-500": 15.60, "BB-Easy": 78.29, "BB-Med": 47.60, "BB-Hard": 22.50, "GAIA": 9.38, "SimpleQA": 28.00, "LoCoMo": 9.62, "LongMemEval": 26.18, "ARC-C": 69.74, "ARC-E": 79.82, "CSQA": 74.63 },
    },
    improvement: 10.94,
  },
  {
    rank: 7, name: "Qwen2.5-3B-Instruct", family: "Qwen", size: "3B", sizeNum: 3,
    effgenAvg: 56.80,
    effgenScores: { "GSM8K": 84.83, "GSM-PLUS": 63.62, "MATH-500": 59.20, "BB-Easy": 81.71, "BB-Med": 35.60, "BB-Hard": 21.67, "GAIA": 15.62, "SimpleQA": 58.00, "LoCoMo": 21.58, "LongMemEval": 32.29, "ARC-C": 86.10, "ARC-E": 93.21, "CSQA": 85.02 },
    frameworkAvgs: { raw: 43.62, langchain: 42.55, autogen: 43.65, smolagents: 31.56 },
    frameworkScores: {
      raw: { "GSM8K": 82.64, "GSM-PLUS": 62.04, "MATH-500": 55.00, "BB-Easy": 42.05, "BB-Med": 18.20, "BB-Hard": 7.33, "GAIA": 6.25, "SimpleQA": 2.00, "LoCoMo": 17.40, "LongMemEval": 29.25, "ARC-C": 79.28, "ARC-E": 90.60, "CSQA": 75.02 },
      langchain: { "GSM8K": 82.83, "GSM-PLUS": 58.33, "MATH-500": 48.80, "BB-Easy": 52.68, "BB-Med": 29.20, "BB-Hard": 5.71, "GAIA": 9.38, "SimpleQA": 12.00, "LoCoMo": 13.42, "LongMemEval": 11.62, "ARC-C": 74.49, "ARC-E": 83.88, "CSQA": 70.84 },
      autogen: { "GSM8K": 79.30, "GSM-PLUS": 60.21, "MATH-500": 51.80, "BB-Easy": 58.78, "BB-Med": 26.80, "BB-Hard": 7.14, "GAIA": 6.25, "SimpleQA": 14.00, "LoCoMo": 11.81, "LongMemEval": 20.57, "ARC-C": 73.04, "ARC-E": 85.14, "CSQA": 72.65 },
      smolagents: { "GSM8K": 69.07, "GSM-PLUS": 49.62, "MATH-500": 10.91, "BB-Easy": 63.41, "BB-Med": 28.80, "BB-Hard": 8.57, "GAIA": 9.38, "SimpleQA": 22.00, "LoCoMo": 7.89, "LongMemEval": 19.54, "ARC-C": 36.69, "ARC-E": 43.43, "CSQA": 41.03 },
    },
    improvement: 13.15,
    timing: { effgen: 17.4, raw: 72.9, langchain: 112.2, autogen: 84.5, smolagents: 126.5 },
  },
  {
    rank: 8, name: "Gemma3-4B", family: "Google", size: "4B", sizeNum: 4,
    effgenAvg: 53.76,
    effgenScores: { "GSM8K": 81.35, "GSM-PLUS": 59.67, "MATH-500": 56.80, "BB-Easy": 78.54, "BB-Med": 33.60, "BB-Hard": 19.58, "GAIA": 12.50, "SimpleQA": 52.00, "LoCoMo": 19.72, "LongMemEval": 29.85, "ARC-C": 82.64, "ARC-E": 91.28, "CSQA": 81.35 },
    frameworkAvgs: { raw: 41.17, langchain: 39.44, autogen: 40.87, smolagents: 29.23 },
    frameworkScores: {
      raw: { "GSM8K": 77.20, "GSM-PLUS": 57.40, "MATH-500": 52.60, "BB-Easy": 40.24, "BB-Med": 17.20, "BB-Hard": 6.67, "GAIA": 6.25, "SimpleQA": 4.00, "LoCoMo": 15.83, "LongMemEval": 26.72, "ARC-C": 72.45, "ARC-E": 87.33, "CSQA": 71.28 },
      langchain: { "GSM8K": 75.48, "GSM-PLUS": 53.92, "MATH-500": 46.40, "BB-Easy": 50.73, "BB-Med": 27.60, "BB-Hard": 5.24, "GAIA": 6.25, "SimpleQA": 10.00, "LoCoMo": 11.84, "LongMemEval": 10.45, "ARC-C": 68.32, "ARC-E": 79.65, "CSQA": 66.82 },
      autogen: { "GSM8K": 74.83, "GSM-PLUS": 55.27, "MATH-500": 49.80, "BB-Easy": 56.10, "BB-Med": 25.20, "BB-Hard": 6.43, "GAIA": 6.25, "SimpleQA": 12.00, "LoCoMo": 10.26, "LongMemEval": 18.63, "ARC-C": 67.18, "ARC-E": 80.92, "CSQA": 68.45 },
      smolagents: { "GSM8K": 65.21, "GSM-PLUS": 47.35, "MATH-500": 10.42, "BB-Easy": 60.98, "BB-Med": 27.20, "BB-Hard": 7.86, "GAIA": 6.25, "SimpleQA": 18.00, "LoCoMo": 6.84, "LongMemEval": 17.54, "ARC-C": 33.48, "ARC-E": 40.25, "CSQA": 38.62 },
    },
    improvement: 12.89,
  },
  {
    rank: 9, name: "Qwen2.5-1.5B-Instruct", family: "Qwen", size: "1.5B", sizeNum: 1.5,
    effgenAvg: 47.44,
    effgenScores: { "GSM8K": 71.63, "GSM-PLUS": 50.25, "MATH-500": 36.00, "BB-Easy": 73.66, "BB-Med": 38.40, "BB-Hard": 7.92, "GAIA": 9.38, "SimpleQA": 40.00, "LoCoMo": 14.19, "LongMemEval": 30.73, "ARC-C": 78.34, "ARC-E": 91.65, "CSQA": 74.62 },
    frameworkAvgs: { raw: 34.28, langchain: 32.81, autogen: 34.33, smolagents: 27.81 },
    frameworkScores: {
      raw: { "GSM8K": 68.01, "GSM-PLUS": 42.83, "MATH-500": 28.60, "BB-Easy": 35.98, "BB-Med": 9.60, "BB-Hard": 2.86, "GAIA": 3.12, "SimpleQA": 4.00, "LoCoMo": 5.13, "LongMemEval": 22.88, "ARC-C": 67.24, "ARC-E": 82.70, "CSQA": 72.73 },
      langchain: { "GSM8K": 66.86, "GSM-PLUS": 37.83, "MATH-500": 23.20, "BB-Easy": 49.39, "BB-Med": 21.60, "BB-Hard": 1.43, "GAIA": 3.12, "SimpleQA": 12.00, "LoCoMo": 8.56, "LongMemEval": 15.34, "ARC-C": 59.13, "ARC-E": 73.78, "CSQA": 54.30 },
      autogen: { "GSM8K": 67.85, "GSM-PLUS": 37.62, "MATH-500": 26.19, "BB-Easy": 52.95, "BB-Med": 28.40, "BB-Hard": 1.67, "GAIA": 6.25, "SimpleQA": 10.00, "LoCoMo": 9.22, "LongMemEval": 18.73, "ARC-C": 53.75, "ARC-E": 65.99, "CSQA": 57.82 },
      smolagents: { "GSM8K": 50.41, "GSM-PLUS": 26.08, "MATH-500": 21.25, "BB-Easy": 56.39, "BB-Med": 30.40, "BB-Hard": 1.43, "GAIA": 3.12, "SimpleQA": 18.00, "LoCoMo": 8.42, "LongMemEval": 22.15, "ARC-C": 53.85, "ARC-E": 37.63, "CSQA": 32.41 },
    },
    improvement: 13.11,
    timing: { effgen: 11.2, raw: 53.7, langchain: 70.5, autogen: 53.7, smolagents: 81.2 },
  },
  {
    rank: 10, name: "Gemma3-1B", family: "Google", size: "1B", sizeNum: 1,
    effgenAvg: 33.38,
    effgenScores: { "GSM8K": 28.28, "GSM-PLUS": 16.04, "MATH-500": 15.40, "BB-Easy": 57.56, "BB-Med": 30.20, "BB-Hard": 6.43, "GAIA": 9.38, "SimpleQA": 32.00, "LoCoMo": 11.54, "LongMemEval": 25.31, "ARC-C": 63.18, "ARC-E": 78.42, "CSQA": 60.15 },
    frameworkAvgs: { raw: 23.11, langchain: 21.88, autogen: 22.51, smolagents: 19.14 },
    frameworkScores: {
      raw: { "GSM8K": 25.80, "GSM-PLUS": 14.60, "MATH-500": 13.00, "BB-Easy": 28.05, "BB-Med": 8.40, "BB-Hard": 2.14, "GAIA": 3.12, "SimpleQA": 3.00, "LoCoMo": 4.82, "LongMemEval": 18.45, "ARC-C": 52.30, "ARC-E": 68.52, "CSQA": 58.24 },
      langchain: { "GSM8K": 24.12, "GSM-PLUS": 12.83, "MATH-500": 10.80, "BB-Easy": 38.54, "BB-Med": 16.80, "BB-Hard": 1.43, "GAIA": 3.12, "SimpleQA": 10.00, "LoCoMo": 6.94, "LongMemEval": 12.68, "ARC-C": 45.21, "ARC-E": 59.84, "CSQA": 42.15 },
      autogen: { "GSM8K": 24.95, "GSM-PLUS": 13.42, "MATH-500": 12.20, "BB-Easy": 41.22, "BB-Med": 22.40, "BB-Hard": 1.67, "GAIA": 6.25, "SimpleQA": 8.00, "LoCoMo": 7.53, "LongMemEval": 15.27, "ARC-C": 41.38, "ARC-E": 52.71, "CSQA": 45.63 },
      smolagents: { "GSM8K": 19.63, "GSM-PLUS": 10.25, "MATH-500": 9.40, "BB-Easy": 44.17, "BB-Med": 24.00, "BB-Hard": 1.43, "GAIA": 3.12, "SimpleQA": 14.00, "LoCoMo": 6.87, "LongMemEval": 17.92, "ARC-C": 41.52, "ARC-E": 30.18, "CSQA": 26.38 },
    },
    improvement: 10.87,
  },
];

const familyColors: Record<string, string> = { Qwen: "#00ff88", Google: "#00e5ff", OpenAI: "#a78bfa" };
const frameworkColors: Record<string, string> = { effgen: "#00ff88", raw: "#6b7280", langchain: "#00e5ff", autogen: "#a78bfa", smolagents: "#ff9500" };
const frameworkLabels: Record<string, string> = { effgen: "effGen", raw: "Raw model", langchain: "LangChain", autogen: "AutoGen", smolagents: "smolagents" };

type ViewTab = "leaderboard" | "comparison" | "speed";
type SortKey = "rank" | "effgenAvg" | "sizeNum" | "improvement";

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) return <span className="text-2xl">{"\u{1F947}"}</span>;
  if (rank === 2) return <span className="text-2xl">{"\u{1F948}"}</span>;
  if (rank === 3) return <span className="text-2xl">{"\u{1F949}"}</span>;
  return <span className="text-lg font-black text-gray-600 dark:text-gray-400 font-mono">#{rank}</span>;
}

function ScoreBar({ value, max, color, delay = 0, animate = true }: { value: number; max: number; color: string; delay?: number; animate?: boolean }) {
  return (
    <div className="h-2 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
      <motion.div
        className="h-full rounded-full"
        style={{ background: `linear-gradient(90deg, ${color}cc, ${color})` }}
        initial={{ width: 0 }}
        animate={animate ? { width: `${(value / max) * 100}%` } : { width: 0 }}
        transition={{ duration: 0.8, delay, ease: "easeOut" }}
      />
    </div>
  );
}

/* ── Expanded benchmark breakdown ── */

function BenchmarkBreakdown({ model, showComparison }: { model: ModelData; showComparison: boolean }) {
  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: "auto", opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="overflow-hidden"
    >
      <div className="px-5 pb-5 md:px-6 md:pb-6">
        <div className="h-px bg-gradient-to-r from-transparent via-gray-300 dark:via-gray-700 to-transparent mb-5" />

        {/* Benchmark scores grid */}
        <div className="space-y-4">
          {benchmarkCategories.map((cat) => (
            <div key={cat.name}>
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full" style={{ background: cat.color }} />
                <span className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-wider">{cat.name}</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {cat.benchmarks.map((b) => {
                  const effgenVal = model.effgenScores[b];
                  const bestBaseline = Math.max(
                    model.frameworkScores.raw[b],
                    model.frameworkScores.langchain[b],
                    model.frameworkScores.autogen[b],
                    model.frameworkScores.smolagents[b],
                  );
                  const delta = effgenVal - bestBaseline;

                  return (
                    <div
                      key={b}
                      className="p-3 rounded-xl bg-gray-50 dark:bg-black/30 border border-gray-200 dark:border-gray-800"
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">{b}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-black" style={accentTextStyle(cat.color)}>{effgenVal.toFixed(1)}%</span>
                          {showComparison && (
                            <span className={`text-[10px] font-bold ${delta >= 0 ? "text-green-700 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                              {delta >= 0 ? "+" : ""}{delta.toFixed(1)}
                            </span>
                          )}
                        </div>
                      </div>
                      <ScoreBar value={effgenVal} max={100} color={cat.color} animate={true} />

                      {/* Framework comparison bars */}
                      {showComparison && (
                        <div className="mt-2 space-y-1">
                          {(["raw", "langchain", "autogen", "smolagents"] as const).map((fw) => (
                            <div key={fw} className="flex items-center gap-2">
                              <span className="text-[9px] w-16 text-gray-600 dark:text-gray-400 truncate">{frameworkLabels[fw]}</span>
                              <div className="flex-1">
                                <ScoreBar value={model.frameworkScores[fw][b]} max={100} color={frameworkColors[fw]} delay={0.1} />
                              </div>
                              <span className="text-[9px] font-mono text-gray-600 dark:text-gray-400 w-10 text-right">{model.frameworkScores[fw][b].toFixed(1)}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/* ── Main Page ── */

export default function LeaderboardView() {
  const [heroRef, heroInView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [tableRef, tableInView] = useInView({ triggerOnce: true, threshold: 0.02 });
  const [activeTab, setActiveTab] = useState<ViewTab>("leaderboard");
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [familyFilter, setFamilyFilter] = useState<string>("all");
  const { version: pypiVersion } = usePyPIVersion();

  const filteredModels = models.filter((m) => {
    if (familyFilter === "all") return true;
    return m.family === familyFilter;
  });

  const sortedModels = [...filteredModels].sort((a, b) => {
    let diff = 0;
    switch (sortKey) {
      case "rank": diff = a.rank - b.rank; break;
      case "effgenAvg": diff = b.effgenAvg - a.effgenAvg; break;
      case "sizeNum": diff = a.sizeNum - b.sizeNum; break;
      case "improvement": diff = b.improvement - a.improvement; break;
    }
    return sortAsc ? diff : -diff;
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const modelsWithTiming = models.filter((m) => m.timing);

  const stats = [
    { value: "10", label: "Models tested", accent: "#00ff88", icon: FiCpu },
    { value: "13", label: "Benchmarks", accent: "#00e5ff", icon: FiBarChart2 },
    { value: "5", label: "Frameworks", accent: "#a78bfa", icon: FiZap },
    { value: "8\u00D7A40", label: "GPUs (46GB)", accent: "#ffd700", icon: FiAward },
  ];

  const viewTabs: { key: ViewTab; label: string; icon: IconType }[] = [
    { key: "leaderboard", label: "Leaderboard", icon: FiAward },
    { key: "comparison", label: "Framework comparison", icon: FiBarChart2 },
    { key: "speed", label: "Speed benchmarks", icon: FiClock },
  ];

  const sortButtons: { key: SortKey; label: string }[] = [
    { key: "rank", label: "Rank" },
    { key: "effgenAvg", label: "Avg score" },
    { key: "sizeNum", label: "Size" },
    { key: "improvement", label: "Improvement" },
  ];

  const families = [
    { key: "all", label: "All families" },
    { key: "Qwen", label: "Qwen" },
    { key: "Google", label: "Gemma" },
    { key: "OpenAI", label: "GPT-OSS" },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">

      {/* Hero */}
      <section className="relative pt-32 pb-16 overflow-hidden">
        <div className="absolute inset-0 grid-pattern" />
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] rounded-full opacity-[0.07] pointer-events-none"
          style={{ background: "radial-gradient(ellipse, #00ff88 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
        <motion.div
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent pointer-events-none z-10"
          animate={{ y: ["-50vh", "50vh"] }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
        />

        <Container className="relative z-10">
          <motion.div
            ref={heroRef}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-center max-w-4xl mx-auto"
          >
            <a
              href="https://arxiv.org/abs/2602.00887"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8 hover:bg-green-500/10 transition-colors"
            >
              <FiAward size={14} />
              arXiv:2602.00887 — under review at ICML 2026
            </a>
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              <span className="gradient-text">Benchmark leaderboard</span>
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-2xl mx-auto">
              10 models &times; 13 benchmarks &times; 5 frameworks &mdash; the evaluation results from the effGen paper
            </p>
          </motion.div>
        </Container>
      </section>

      {/* Stats Banner */}
      <section className="py-6 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <Container className="relative z-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {stats.map((stat, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={heroInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                className="relative overflow-hidden rounded-2xl p-5 bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 text-center backdrop-blur-sm"
              >
                <div className="absolute inset-0 opacity-[0.04]" style={{ background: `radial-gradient(circle at 50% 50%, ${stat.accent} 0%, transparent 70%)` }} />
                <stat.icon className="mx-auto mb-2" style={accentTextStyle(stat.accent)} size={22} />
                <div className="text-2xl font-black mb-0.5" style={accentTextStyle(stat.accent)}>{stat.value}</div>
                <div className="text-[10px] text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </Container>
      </section>

      {/* View Tabs */}
      <section className="py-4 relative">
        <Container className="relative z-10">
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {viewTabs.map((tab) => (
              <motion.button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-bold transition-all border"
                style={{
                  background: activeTab === tab.key ? "rgba(0,255,136,0.15)" : "transparent",
                  borderColor: activeTab === tab.key ? "rgba(0,255,136,0.5)" : "rgba(128,128,128,0.3)",
                  ...(activeTab === tab.key ? accentTextStyle("#00ff88") : {}),
                  boxShadow: activeTab === tab.key ? "0 0 20px rgba(0,255,136,0.15)" : "none",
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <tab.icon size={14} className={activeTab !== tab.key ? "text-gray-600 dark:text-gray-400" : ""} />
                <span className={activeTab !== tab.key ? "text-gray-600 dark:text-gray-400" : ""}>{tab.label}</span>
              </motion.button>
            ))}
          </div>

          {/* Controls (sort + filter) */}
          {activeTab !== "speed" && (
            <div className="flex flex-wrap gap-4 justify-center items-center">
              {/* Family filter */}
              <div className="flex flex-wrap gap-2">
                {families.map((f) => (
                  <motion.button
                    key={f.key}
                    onClick={() => setFamilyFilter(f.key)}
                    className="px-3 py-1.5 rounded-full text-xs font-bold border transition-all"
                    style={{
                      background: familyFilter === f.key ? `${familyColors[f.key] || "#00ff88"}15` : "transparent",
                      borderColor: familyFilter === f.key ? `${familyColors[f.key] || "#00ff88"}50` : "rgba(128,128,128,0.2)",
                      ...(familyFilter === f.key ? accentTextStyle(familyColors[f.key] || "#00ff88") : {}),
                    }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <span className={familyFilter !== f.key ? "text-gray-600 dark:text-gray-400" : ""}>{f.label}</span>
                  </motion.button>
                ))}
              </div>

              <div className="w-px h-5 bg-gray-300 dark:bg-gray-700 hidden md:block" />

              {/* Sort controls */}
              <div className="flex flex-wrap gap-2 items-center">
                <span className="text-[10px] text-gray-600 dark:text-gray-400 uppercase tracking-wider font-semibold mr-1">Sort:</span>
                {sortButtons.map((btn) => (
                  <motion.button
                    key={btn.key}
                    onClick={() => handleSort(btn.key)}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-xl text-xs font-bold border transition-all"
                    style={{
                      background: sortKey === btn.key ? "rgba(0,255,136,0.1)" : "transparent",
                      borderColor: sortKey === btn.key ? "rgba(0,255,136,0.3)" : "rgba(128,128,128,0.2)",
                      ...(sortKey === btn.key ? accentTextStyle("#00ff88") : {}),
                    }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <span className={sortKey !== btn.key ? "text-gray-600 dark:text-gray-400" : ""}>{btn.label}</span>
                    {sortKey === btn.key && (
                      <motion.span animate={{ rotate: sortAsc ? 0 : 180 }} transition={{ duration: 0.2 }}>
                        <FiArrowUp size={10} />
                      </motion.span>
                    )}
                  </motion.button>
                ))}
              </div>
            </div>
          )}
        </Container>
      </section>

      {/* ══ LEADERBOARD VIEW ══ */}
      {(activeTab === "leaderboard" || activeTab === "comparison") && (
        <section className="py-6 pb-20 relative">
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10">
            <h2 className="sr-only">Model rankings</h2>
            <motion.div
              ref={tableRef}
              initial="hidden"
              animate={tableInView ? "visible" : "hidden"}
              variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.05 } } }}
              className="space-y-3"
            >
              <AnimatePresence mode="popLayout">
                {sortedModels.map((model) => {
                  const familyColor = familyColors[model.family] || "#00ff88";
                  const isTop3 = model.rank <= 3;
                  const isExpanded = expandedModel === model.name;
                  const bestBaseline = Math.max(model.frameworkAvgs.raw, model.frameworkAvgs.langchain, model.frameworkAvgs.autogen, model.frameworkAvgs.smolagents);

                  return (
                    <motion.div
                      key={model.name}
                      layout
                      variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="group relative overflow-hidden rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm"
                    >
                      {isTop3 && (
                        <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ background: `linear-gradient(135deg, ${familyColor} 0%, transparent 60%)` }} />
                      )}

                      <motion.div
                        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl pointer-events-none"
                        style={{ boxShadow: `inset 0 0 0 1px ${familyColor}30, 0 0 30px ${familyColor}08` }}
                      />

                      {/* Main row */}
                      <div
                        className="p-5 md:p-6 relative z-10 cursor-pointer"
                        onClick={() => setExpandedModel(isExpanded ? null : model.name)}
                      >
                        <div className="flex flex-col md:flex-row md:items-center gap-4 md:gap-6">
                          {/* Rank */}
                          <div className="flex items-center gap-4 md:w-12 flex-shrink-0">
                            <RankBadge rank={model.rank} />
                          </div>

                          {/* Model info */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <h3 className="text-base font-black text-gray-900 dark:text-white truncate">{model.name}</h3>
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold border" style={accentTextStyle(familyColor, { background: `${familyColor}10`, borderColor: `${familyColor}30` })}>
                                {model.family}
                              </span>
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
                                {model.size}
                              </span>
                            </div>
                          </div>

                          {/* Scores */}
                          <div className="flex items-center gap-5 md:gap-8 flex-shrink-0">
                            {/* effGen average */}
                            <div className="text-center">
                              <div className="text-2xl font-black" style={accentTextStyle("#00ff88")}>
                                {model.effgenAvg.toFixed(2)}
                              </div>
                              <div className="text-[9px] text-gray-600 dark:text-gray-400 uppercase tracking-wider font-semibold">effGen avg</div>
                            </div>

                            {/* Score bar */}
                            <div className="hidden sm:block w-28 md:w-36">
                              <ScoreBar value={model.effgenAvg} max={75} color="#00ff88" delay={model.rank * 0.05} animate={tableInView} />
                              {activeTab === "comparison" && (
                                <div className="mt-1">
                                  <ScoreBar value={bestBaseline} max={75} color="#6b7280" delay={model.rank * 0.05 + 0.2} animate={tableInView} />
                                </div>
                              )}
                            </div>

                            {/* Improvement */}
                            <div className="text-center">
                              <div className="text-lg font-black text-green-700 dark:text-green-400">+{model.improvement.toFixed(1)}</div>
                              <div className="text-[9px] text-gray-600 dark:text-gray-400 uppercase tracking-wider font-semibold">vs Best</div>
                            </div>

                            {/* Expand chevron */}
                            <motion.div
                              animate={{ rotate: isExpanded ? 180 : 0 }}
                              transition={{ duration: 0.2 }}
                              className="text-gray-400"
                            >
                              <FiChevronDown size={18} />
                            </motion.div>
                          </div>
                        </div>

                        {/* Framework averages strip (comparison view) */}
                        {activeTab === "comparison" && (
                          <div className="mt-3 flex flex-wrap gap-3">
                            {(["effgen", "autogen", "langchain", "smolagents", "raw"] as const).map((fw) => {
                              const val = fw === "effgen" ? model.effgenAvg : model.frameworkAvgs[fw];
                              return (
                                <div key={fw} className="flex items-center gap-1.5">
                                  <span className="w-2 h-2 rounded-full" style={{ background: frameworkColors[fw] }} />
                                  <span className="text-[10px] text-gray-600 dark:text-gray-400">{frameworkLabels[fw]}</span>
                                  <span className="text-[10px] font-bold" style={accentTextStyle(frameworkColors[fw])}>{val.toFixed(1)}</span>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>

                      {/* Expanded benchmark breakdown */}
                      <AnimatePresence>
                        {isExpanded && (
                          <BenchmarkBreakdown model={model} showComparison={activeTab === "comparison"} />
                        )}
                      </AnimatePresence>

                      <div className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: `linear-gradient(90deg, transparent, ${familyColor}, transparent)` }} />
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </motion.div>
          </Container>
        </section>
      )}

      {/* ══ SPEED BENCHMARKS VIEW ══ */}
      {activeTab === "speed" && (
        <section className="py-6 pb-20 relative">
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10 max-w-5xl">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="mb-8 text-center"
            >
              <p className="text-sm text-gray-600 dark:text-gray-400">Average execution time (minutes) across 13 benchmarks. Lower is better. Qwen2.5 family shown.</p>
            </motion.div>

            <div className="space-y-4">
              {modelsWithTiming.map((model, idx) => {
                const t = model.timing!;
                const maxTime = Math.max(t.raw, t.langchain, t.autogen, t.smolagents);
                const speedup = (Math.min(t.raw, t.langchain, t.autogen, t.smolagents) / t.effgen).toFixed(1);

                return (
                  <motion.div
                    key={model.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="group relative overflow-hidden rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 p-5 md:p-6"
                  >
                    <motion.div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl pointer-events-none" style={{ boxShadow: "inset 0 0 0 1px rgba(0,255,136,0.2)" }} />

                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-black text-gray-900 dark:text-white">{model.name}</span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{model.size}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <FiZap className="text-green-700 dark:text-green-400" size={14} />
                        <span className="text-sm font-black text-green-700 dark:text-green-400">{speedup}x faster</span>
                      </div>
                    </div>

                    {/* Timing bars */}
                    <div className="space-y-2">
                      {(["effgen", "raw", "autogen", "langchain", "smolagents"] as const).map((fw) => {
                        const val = fw === "effgen" ? t.effgen : t[fw];
                        const isEffgen = fw === "effgen";
                        return (
                          <div key={fw} className="flex items-center gap-3">
                            <span className="text-[10px] font-semibold w-20 text-right" style={accentTextStyle(isEffgen ? "#00ff88" : "#6b7280")}>
                              {frameworkLabels[fw]}
                            </span>
                            <div className="flex-1">
                              <div className="h-3 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
                                <motion.div
                                  className="h-full rounded-full"
                                  style={{ background: isEffgen ? "linear-gradient(90deg, #00ff88cc, #00ff88)" : `${frameworkColors[fw]}80` }}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${(val / maxTime) * 100}%` }}
                                  transition={{ duration: 0.8, delay: 0.1 }}
                                />
                              </div>
                            </div>
                            <span className="text-[10px] font-mono w-14 text-right" style={accentTextStyle(isEffgen ? "#00ff88" : "#9ca3af")}>
                              {val.toFixed(1)} min
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </Container>
        </section>
      )}

      {/* Testing Environment */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              viewport={{ once: true }}
              className="relative overflow-hidden rounded-2xl bg-white dark:bg-gray-900/80 border border-gray-200 dark:border-green-500/20 shadow-sm dark:shadow-none p-8 md:p-10"
              style={{ boxShadow: "0 0 40px rgba(0,255,136,0.04)" }}
            >
              <motion.div className="absolute top-0 left-0 right-0 h-px" style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }} animate={{ opacity: [0.3, 0.8, 0.3] }} transition={{ duration: 3, repeat: Infinity }} />
              <div className="absolute inset-0 grid-pattern opacity-30" />

              <div className="relative z-10">
                <div className="flex items-center gap-3 mb-6">
                  <motion.div
                    className="w-12 h-12 rounded-xl flex items-center justify-center"
                    style={{ background: "rgba(0,255,136,0.1)", border: "1px solid rgba(0,255,136,0.25)" }}
                    whileHover={{ rotate: 10, scale: 1.1 }}
                  >
                    <FiCpu className="text-green-700 dark:text-green-400" size={22} />
                  </motion.div>
                  <div>
                    <h2 className="text-2xl font-black text-gray-900 dark:text-white">Evaluation Setup</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">From arXiv:2602.00887, under review at ICML 2026</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-5 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-green-700 dark:text-green-400 uppercase tracking-wider font-bold mb-2">Hardware</div>
                    <div className="text-base font-bold text-gray-900 dark:text-white font-mono">8&times; NVIDIA A40 (46GB VRAM)</div>
                  </div>

                  <div className="p-5 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-violet-600 dark:text-violet-400 uppercase tracking-wider font-bold mb-2">Software</div>
                    <div className="flex flex-wrap gap-2 font-mono text-sm text-gray-700 dark:text-gray-300">
                      <span>Python 3.11</span>
                      <span className="text-gray-400">|</span>
                      <span>vLLM, local weights</span>
                    </div>
                    <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 font-sans leading-relaxed">
                      Every model ran on this machine, through each framework in turn. The scores
                      are the paper&rsquo;s, not a re-run: they were measured at the version the
                      paper reports, which is not necessarily the {pypiVersion} on PyPI today.
                    </p>
                  </div>

                  <div className="p-5 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-cyan-600 dark:text-cyan-400 uppercase tracking-wider font-bold mb-3">13 Benchmarks</div>
                    <div className="flex flex-wrap gap-1.5">
                      {allBenchmarks.map((b) => {
                        const cat = getBenchmarkCategory(b);
                        return (
                          <span key={b} className="px-2 py-1 rounded-lg text-[10px] font-semibold border" style={{ background: `${cat.color}10`, color: cat.color, borderColor: `${cat.color}25` }}>
                            {b}
                          </span>
                        );
                      })}
                    </div>
                  </div>

                  <div className="p-5 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-orange-700 dark:text-orange-400 uppercase tracking-wider font-bold mb-3">Frameworks Compared</div>
                    <div className="flex flex-wrap gap-1.5">
                      {(["effgen", "langchain", "autogen", "smolagents", "raw"] as const).map((fw) => (
                        <span key={fw} className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10px] font-semibold bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                          <span className="w-1.5 h-1.5 rounded-full" style={{ background: frameworkColors[fw] }} />
                          <span className="text-gray-600 dark:text-gray-400">{frameworkLabels[fw]}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </Container>
      </section>

      {/* Key Finding */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="relative overflow-hidden rounded-2xl p-10 text-center max-w-3xl mx-auto bg-gray-50 dark:bg-gray-950/90 border border-gray-200 dark:border-green-500/15 shadow-sm dark:shadow-none"
            style={{ boxShadow: "0 0 50px rgba(0,255,136,0.05)" }}
          >
            <motion.div className="absolute top-0 left-0 right-0 h-px" style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }} animate={{ opacity: [0.3, 0.8, 0.3] }} transition={{ duration: 3, repeat: Infinity }} />
            <div className="absolute inset-0 grid-pattern opacity-50" />

            <div className="relative z-10">
              <motion.div className="mb-5 inline-block" animate={{ y: [0, -8, 0] }} transition={{ duration: 2, repeat: Infinity }}>
                <FiTrendingUp size={48} className="text-green-700 dark:text-green-400 mx-auto" />
              </motion.div>
              <h2 className="text-2xl md:text-3xl font-black mb-3 text-gray-900 dark:text-white">
                <span className="gradient-text">Key finding</span>
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-xl mx-auto text-sm leading-relaxed">
                effGen scores above LangChain, AutoGen and smolagents on all 10 models and all 13 benchmarks, with the largest gains on the smaller models.
              </p>
              <div className="flex flex-wrap gap-3 justify-center">
                <div className="px-4 py-2 rounded-xl bg-green-500/10 border border-green-500/25 text-green-700 dark:text-green-400 text-sm font-bold">
                  +13.15% avg gain (3B)
                </div>
                <div className="px-4 py-2 rounded-xl bg-cyan-500/10 border border-cyan-500/25 text-cyan-600 dark:text-cyan-400 text-sm font-bold">
                  Up to 7.7x faster
                </div>
                <div className="px-4 py-2 rounded-xl bg-violet-500/10 border border-violet-500/25 text-violet-600 dark:text-violet-400 text-sm font-bold">
                  70.97% top score (32B)
                </div>
              </div>
            </div>
          </motion.div>
        </Container>
      </section>

      </main>
      <Footer />
    </div>
  );
}
