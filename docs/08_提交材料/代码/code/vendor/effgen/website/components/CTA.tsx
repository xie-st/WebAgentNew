"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiArrowRight, FiBook, FiZap, FiStar, FiGitBranch, FiUsers } from "react-icons/fi";
import Container from "./Container";
import { useGitHubStats } from "./GitHubStats";
import { withBasePath } from "./basePath";
import { version } from "./siteData";
import { accentTextStyle } from "./accentText";

// Floating geometric shapes positions (fixed for SSR)
const SHAPES = [
  { type: "hexagon", x: "10%", y: "20%", size: 24, duration: 12, delay: 0, rotation: 45 },
  { type: "triangle", x: "85%", y: "15%", size: 20, duration: 15, delay: 2, rotation: 0 },
  { type: "hexagon", x: "90%", y: "70%", size: 18, duration: 10, delay: 1, rotation: 30 },
  { type: "triangle", x: "5%", y: "75%", size: 22, duration: 14, delay: 3, rotation: 60 },
  { type: "hexagon", x: "50%", y: "85%", size: 16, duration: 11, delay: 0.5, rotation: 15 },
];

function FloatingShape({ shape }: { shape: typeof SHAPES[0] }) {
  const pathData = shape.type === "hexagon"
    ? "M12 0L22 6.5V17.5L12 24L2 17.5V6.5Z"
    : "M12 0L24 24H0Z";

  return (
    <motion.div
      className="absolute pointer-events-none"
      style={{ left: shape.x, top: shape.y }}
      animate={{
        y: [0, -20, 0],
        rotate: [shape.rotation, shape.rotation + 180, shape.rotation + 360],
      }}
      transition={{
        duration: shape.duration,
        delay: shape.delay,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    >
      <svg width={shape.size} height={shape.size} viewBox="0 0 24 24" fill="none" className="opacity-[0.06]">
        <path d={pathData} stroke="#00ff88" strokeWidth="1" />
      </svg>
    </motion.div>
  );
}

export default function CTA() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });
  const githubStats = useGitHubStats();

  const formatNumber = (num: number): string => {
    if (num >= 1000) return (num / 1000).toFixed(1) + "k";
    return num.toString();
  };

  const ctaStats = [
    { icon: FiStar, value: githubStats.loading ? "..." : formatNumber(githubStats.stars), label: "GitHub stars", color: "#ffd700" },
    { icon: FiGitBranch, value: githubStats.loading ? "..." : formatNumber(githubStats.forks), label: "Forks", color: "#00e5ff" },
    { icon: FiUsers, value: githubStats.loading ? "..." : githubStats.contributors.toString(), label: "Contributors", color: "#a78bfa" },
  ];

  // Words for word-by-word reveal
  const headingWords = ["One", "install,", "and", "an"];
  const headingWords2 = ["agent", "that", "runs"];

  return (
    <section className="py-24 bg-gray-50 dark:bg-[#030f07] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      {/* Animated gradient mesh background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <motion.div
          className="absolute w-[600px] h-[600px] rounded-full opacity-[0.04]"
          style={{ background: "radial-gradient(circle, #00ff88 0%, transparent 70%)", left: "20%", top: "10%" }}
          animate={{ x: [0, 80, 0], y: [0, -40, 0] }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute w-[500px] h-[500px] rounded-full opacity-[0.03]"
          style={{ background: "radial-gradient(circle, #00e5ff 0%, transparent 70%)", right: "10%", bottom: "10%" }}
          animate={{ x: [0, -60, 0], y: [0, 60, 0] }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute w-[400px] h-[400px] rounded-full opacity-[0.03]"
          style={{ background: "radial-gradient(circle, #a78bfa 0%, transparent 70%)", left: "50%", top: "50%", marginLeft: "-200px", marginTop: "-200px" }}
          animate={{ scale: [1, 1.3, 1], opacity: [0.03, 0.06, 0.03] }}
          transition={{ duration: 8, repeat: Infinity }}
        />
      </div>

      {/* Floating geometric shapes */}
      {SHAPES.map((shape, i) => (
        <FloatingShape key={i} shape={shape} />
      ))}

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="relative overflow-hidden rounded-2xl border border-green-500/20 bg-white dark:bg-gray-950/80 backdrop-blur-sm noise-overlay"
          style={{ boxShadow: "0 0 80px rgba(0,255,136,0.08)" }}
        >
          {/* Animated top border */}
          <motion.div
            className="absolute top-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, #00ff88, #00c96e, transparent)" }}
            animate={{
              background: [
                "linear-gradient(90deg, transparent, #00ff88, #00c96e, transparent)",
                "linear-gradient(90deg, transparent, #00c96e, #00ff88, transparent)",
              ],
            }}
            transition={{ duration: 3, repeat: Infinity }}
          />

          {/* Grid overlay */}
          <div className="absolute inset-0 grid-pattern opacity-50" />

          {/* Floating orbs */}
          <motion.div
            className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-[0.06] pointer-events-none"
            style={{ background: "radial-gradient(circle, #00ff88 0%, transparent 70%)" }}
            animate={{ scale: [1, 1.3, 1], y: [0, -20, 0] }}
            transition={{ duration: 6, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-0 left-0 w-64 h-64 rounded-full opacity-[0.06] pointer-events-none"
            style={{ background: "radial-gradient(circle, #00c96e 0%, transparent 70%)" }}
            animate={{ scale: [1.3, 1, 1.3], y: [0, 20, 0] }}
            transition={{ duration: 8, repeat: Infinity }}
          />

          <div className="relative z-10 text-center py-16 px-8 lg:px-20">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8"
            >
              <motion.div
                className="w-2 h-2 rounded-full bg-green-400"
                animate={{ scale: [1, 1.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              effGen {version} — Apache 2.0
            </motion.div>

            {/* Word-by-word reveal heading */}
            <h2 className="text-4xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              {headingWords.map((word, i) => (
                <motion.span
                  key={i}
                  className="inline-block mr-[0.3em]"
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.3 + i * 0.1 }}
                >
                  {word}
                </motion.span>
              ))}
              <br />
              {headingWords2.map((word, i) => (
                <motion.span
                  key={i}
                  className={`inline-block mr-[0.3em] ${i < headingWords2.length ? "gradient-text neon-text" : ""}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.7 + i * 0.1 }}
                >
                  {word}
                </motion.span>
              ))}
            </h2>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="text-lg text-gray-600 dark:text-gray-400 mb-10 max-w-2xl mx-auto"
            >
              <code className="font-mono">pip install effgen</code>, then one line to build an
              agent. Run it against a model on your own hardware, a server you already have, or any
              of the providers effGen ships an adapter for.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12"
            >
              {/* Primary CTA with continuous pulse */}
              <motion.a
                href="#quickstart"
                className="group relative px-10 py-5 rounded-full font-bold text-black text-lg flex items-center gap-2 overflow-hidden"
                style={{
                  background: "linear-gradient(135deg, #00ff88, #00c96e)",
                }}
                animate={{
                  boxShadow: [
                    "0 0 30px rgba(0,255,136,0.3)",
                    "0 0 50px rgba(0,255,136,0.5)",
                    "0 0 30px rgba(0,255,136,0.3)",
                  ],
                }}
                transition={{ duration: 2, repeat: Infinity }}
                whileHover={{ scale: 1.05, y: -2, boxShadow: "0 0 60px rgba(0,255,136,0.6)" }}
                whileTap={{ scale: 0.95 }}
              >
                <span className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition-colors" />
                <FiZap />
                Quick start
                <FiArrowRight className="group-hover:translate-x-1 transition-transform" />
              </motion.a>

              <motion.a
                href={withBasePath("/docs")}
                className="px-10 py-5 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/50 text-lg flex items-center gap-2 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiBook size={18} />
                Read the documentation
              </motion.a>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="grid grid-cols-3 gap-4 max-w-lg mx-auto"
            >
              {ctaStats.map((stat, index) => (
                <motion.div
                  key={index}
                  className="text-center p-4 rounded-xl bg-gray-50 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-800"
                  whileHover={{ scale: 1.05, y: -3, borderColor: stat.color + "40" }}
                >
                  <div className="flex justify-center mb-2">
                    <stat.icon style={accentTextStyle(stat.color)} size={18} />
                  </div>
                  <div className="text-2xl font-black mb-1" style={accentTextStyle(stat.color)}>{stat.value}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 font-medium">{stat.label}</div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </motion.div>
      </Container>
    </section>
  );
}
