"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiGithub, FiTwitter, FiArrowRight, FiUsers, FiCode, FiBookOpen, FiAlertCircle, FiAward } from "react-icons/fi";
import { SiDiscord } from "react-icons/si";
import { FaLinkedin as SiLinkedin } from "react-icons/fa6";
import Link from "next/link";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useGitHubStats } from "@/components/GitHubStats";
import { accentTextStyle } from "@/components/accentText";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

export default function CommunityView() {
  const githubStats = useGitHubStats();
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  const fmt = (n: number) => (n >= 1000 ? (n / 1000).toFixed(1) + "k" : n.toString());

  // Only GitHub carries figures, because the GitHub API reports them and this
  // page reads them. The other three used to show tiles reading "New" and
  // "Active" where a follower count would go; nothing here can read those
  // numbers, so the tiles are gone rather than filled with a placeholder.
  const platforms = [
    {
      name: "GitHub",
      icon: FiGithub,
      description: "The source, the issue tracker and the releases. Star it, file a bug, or open a pull request.",
      stats: [
        { label: "Stars", value: githubStats.loading ? "…" : fmt(githubStats.stars) },
        { label: "Contributors", value: githubStats.loading ? "…" : githubStats.contributors.toString() },
      ],
      link: "https://github.com/ctrl-gaurav/effGen",
      linkText: "Open the repository",
      accent: "#e8eaed",
    },
    {
      name: "Discord",
      icon: SiDiscord,
      description: "Ask a question, show what you built, or work through a problem with someone who has hit it.",
      stats: [],
      link: "https://discord.com/invite/jacn9ed3",
      linkText: "Join the server",
      accent: "#5865f2",
    },
    {
      name: "LinkedIn",
      icon: SiLinkedin,
      description: "Release announcements and project updates, for anyone who follows their tools there.",
      stats: [],
      link: "https://www.linkedin.com/company/111341317/",
      linkText: "Follow the page",
      accent: "#0a66c2",
    },
    {
      name: "X",
      icon: FiTwitter,
      description: "Short-form release notes and links to what is new.",
      stats: [],
      link: "https://x.com/effGen_org",
      linkText: "Follow the account",
      accent: "#1d9bf0",
    },
  ];

  const contributions = [
    { icon: FiCode, emoji: "💻", title: "Code", desc: "A tool, a provider adapter, a preset, a fix. The contributing guide covers the setup, the style and what a pull request needs.", accent: "#00ff88" },
    { icon: FiBookOpen, emoji: "📚", title: "Documentation", desc: "A page that is wrong, a step that is missing, or a sample that no longer runs against the released package.", accent: "#00e5ff" },
    { icon: FiAward, emoji: "🎨", title: "Examples", desc: "A short program that shows the framework doing something the existing examples do not.", accent: "#a78bfa" },
    { icon: FiAlertCircle, emoji: "🐛", title: "Issues", desc: "A bug, with the command and the version that reproduce it — or a feature, with the problem behind it.", accent: "#ff6b6b" },
  ];

  const guidelines = [
    { num: "01", title: "Be respectful", desc: "Harassment, trolling and personal attacks are not tolerated." },
    { num: "02", title: "Be helpful", desc: "Share what you know, give feedback someone can act on, and answer the people who are new." },
    { num: "03", title: "Stay on topic", desc: "Keep discussions about effGen and building with it. Off-topic posts may be removed." },
    { num: "04", title: "Show your work", desc: "A report carrying the command, the version and the output is one somebody can act on." },
    { num: "05", title: "Give credit", desc: "Credit the author and the source of anything you bring in." },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">

      {/* Hero */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0 grid-pattern" />
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] rounded-full opacity-[0.06] pointer-events-none"
          style={{ background: "radial-gradient(ellipse, #00ff88 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
        {/* Scan line */}
        <motion.div
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent pointer-events-none z-10"
          animate={{ y: ["-50vh", "50vh"] }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
        />

        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-center max-w-4xl mx-auto"
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8"
              whileHover={{ borderColor: "rgba(0,255,136,0.6)" }}
            >
              <FiUsers size={14} />
              Community
            </motion.div>

            <h1 className="text-5xl md:text-6xl lg:text-7xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              Where effGen is{" "}
              <span className="gradient-text neon-text">worked on</span>
            </h1>

            <p className="text-xl text-gray-600 dark:text-gray-400 leading-relaxed">
              The repository, the issue tracker, the chat, and how to cite the work if you build
              on it.
            </p>
          </motion.div>
        </Container>
      </section>

      {/* Community Platforms */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <h2 className="sr-only">Where the community is</h2>
          <motion.div
            ref={ref}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5"
          >
            {platforms.map((platform, idx) => (
              <motion.div
                key={platform.name}
                variants={{ hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } }}
                whileHover={{ y: -8, scale: 1.02 }}
                transition={{ duration: 0.3 }}
                className="group relative overflow-hidden p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm"
              >
                {/* Animated rotating border */}
                <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                  <motion.div
                    className="absolute inset-[-50%] w-[200%] h-[200%]"
                    style={{ background: `conic-gradient(from 0deg, transparent 60%, ${platform.accent}25 80%, transparent 100%)` }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                </div>

                {/* Hover glow */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                  style={{ background: `radial-gradient(circle at 30% 30%, ${platform.accent}10 0%, transparent 70%)` }}
                />
                <motion.div
                  className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ boxShadow: `inset 0 0 0 1px ${platform.accent}30` }}
                />

                {/* Icon */}
                <motion.div
                  className="w-14 h-14 rounded-xl flex items-center justify-center mb-5 relative z-10"
                  style={{ background: `${platform.accent}15`, border: `1px solid ${platform.accent}25` }}
                  whileHover={{ rotate: 10, scale: 1.1 }}
                >
                  <platform.icon style={accentTextStyle(platform.accent)} size={24} aria-hidden="true" />
                </motion.div>

                {/* Name with pulsing status dot */}
                <div className="flex items-center gap-2 mb-2 relative z-10">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">{platform.name}</h3>
                  <motion.div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: platform.accent, boxShadow: `0 0 8px ${platform.accent}` }}
                    animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
                    transition={{ duration: 2, repeat: Infinity, delay: idx * 0.3 }}
                  />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-5 leading-relaxed relative z-10">{platform.description}</p>

                {platform.stats.length > 0 && (
                  <div className="grid grid-cols-2 gap-3 mb-5 relative z-10">
                    {platform.stats.map((stat) => (
                      <div key={stat.label} className="text-center p-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                        <div className="text-xl font-black" style={accentTextStyle(platform.accent)}>{stat.value}</div>
                        <div className="text-[10px] text-gray-600 dark:text-gray-400 font-medium uppercase tracking-wide">{stat.label}</div>
                      </div>
                    ))}
                  </div>
                )}

                <motion.a
                  href={platform.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between p-3 rounded-xl font-semibold text-sm relative z-10 transition-all"
                  style={accentTextStyle(platform.accent, { background: `${platform.accent}15`, border: `1px solid ${platform.accent}30` })}
                  whileHover={{ backgroundColor: `${platform.accent}25` }}
                  whileTap={{ scale: 0.97 }}
                >
                  <span>{platform.linkText}</span>
                  <FiArrowRight size={14} />
                </motion.a>

                <div
                  className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(90deg, transparent, ${platform.accent}60, transparent)` }}
                />
              </motion.div>
            ))}
          </motion.div>
        </Container>
      </section>

      {/* Contribute */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto text-center"
          >
            <motion.div
              className="text-5xl mb-6 inline-block"
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              🚀
            </motion.div>

            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
              <span className="gradient-text">Contribute to effGen</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-12 leading-relaxed max-w-2xl mx-auto">
              Code, documentation, examples and bug reports are all wanted. The contributing guide
              covers the development setup, the test lanes and what a pull request needs before it
              can be merged.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
              {contributions.map((c, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -6, scale: 1.02 }}
                  className="group relative p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none text-left overflow-hidden"
                >
                  {/* Animated rotating border */}
                  <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                    <motion.div
                      className="absolute inset-[-50%] w-[200%] h-[200%]"
                      style={{ background: `conic-gradient(from 0deg, transparent 60%, ${c.accent}25 80%, transparent 100%)` }}
                      animate={{ rotate: 360 }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    />
                    <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                  </div>

                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{ background: `radial-gradient(circle at 20% 50%, ${c.accent}08 0%, transparent 70%)` }}
                  />
                  <motion.div
                    className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 text-xl relative z-10"
                    style={{ background: `${c.accent}15`, border: `1px solid ${c.accent}25` }}
                    whileHover={{ rotate: 360 }}
                    transition={{ duration: 0.5 }}
                  >
                    {c.emoji}
                  </motion.div>
                  <h3 className="text-base font-bold mb-2 text-gray-900 dark:text-white relative z-10">{c.title}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 relative z-10">{c.desc}</p>
                  <div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${c.accent}50, transparent)` }}
                  />
                </motion.div>
              ))}
            </div>

            <div className="flex flex-wrap gap-4 justify-center">
              <motion.a
                href="https://github.com/ctrl-gaurav/effGen/blob/main/CONTRIBUTING.md"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)", boxShadow: "0 0 25px rgba(0,255,136,0.3)" }}
                whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,136,0.5)" }}
                whileTap={{ scale: 0.95 }}
              >
                Contributing Guide
                <FiArrowRight size={16} />
              </motion.a>
              <motion.a
                href="https://github.com/ctrl-gaurav/effGen/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                View Issues
              </motion.a>
            </div>
          </motion.div>
        </Container>
      </section>

      {/* Community Guidelines */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-4xl font-black mb-10 text-center text-gray-900 dark:text-white">Community guidelines</h2>

            <div className="rounded-2xl bg-white dark:bg-gray-900/80 border border-gray-200 dark:border-green-500/20 shadow-sm dark:shadow-none p-8 space-y-6" style={{ boxShadow: "0 0 40px rgba(0,255,136,0.04)" }}>
              {guidelines.map((g, i) => (
                <motion.div
                  key={i}
                  className="flex items-start gap-4"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: i * 0.08 }}
                  viewport={{ once: true }}
                >
                  <motion.div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 font-black text-sm font-mono"
                    style={{ background: "rgba(0,255,136,0.1)", border: "1px solid rgba(0,255,136,0.25)", color: "#00ff88" }}
                    whileHover={{ scale: 1.15, boxShadow: "0 0 20px rgba(0,255,136,0.3)" }}
                    transition={{ duration: 0.3 }}
                  >
                    {g.num}
                  </motion.div>
                  <div>
                    <h3 className="text-base font-bold mb-1 text-gray-900 dark:text-white">{g.title}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{g.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </Container>
      </section>

      {/* Cite it, and the licence */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto grid lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] gap-6 items-start"
          >
            <div className="rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 p-8">
              <h2 className="flex items-center gap-2 text-2xl font-black mb-3 text-gray-900 dark:text-white">
                <FiBookOpen size={20} className="text-green-500" aria-hidden="true" />
                Citing effGen
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
                If effGen is part of your research, cite the paper. It is on arXiv as{" "}
                <a
                  href="https://arxiv.org/abs/2602.00887"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-700 dark:text-green-400 font-semibold"
                >
                  arXiv:2602.00887
                </a>
                , and the benchmark results behind it are on the{" "}
                <Link href="/leaderboard" className="text-green-700 dark:text-green-400 font-semibold">
                  leaderboard
                </Link>
                .
              </p>
              <pre
                className="p-4 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 overflow-x-auto text-xs font-mono text-gray-700 dark:text-gray-300 leading-relaxed focus-visible:outline focus-visible:outline-2 focus-visible:outline-green-500 focus-visible:-outline-offset-2"
                tabIndex={0}
                role="group"
                aria-label="Citation"
              >
{`@software{srivastava2026effgen,
      title={effGen: Enabling Small Language Models as Capable Autonomous Agents},
      author={Gaurav Srivastava and Aafiya Hussain and Chi Wang and
              Yingyan Celine Lin and Xuan Wang},
      year={2026},
      eprint={2602.00887},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.00887},
}`}
              </pre>
            </div>

            <div className="rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 p-8">
              <h2 className="flex items-center gap-2 text-2xl font-black mb-3 text-gray-900 dark:text-white">
                <FiAward size={20} className="text-green-500" aria-hidden="true" />
                Licence
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-4">
                effGen is released under the <strong>Apache&nbsp;2.0</strong> licence: use it
                commercially, modify it and redistribute it, with a patent grant, provided you keep
                the notice and state what you changed.
              </p>
              <a
                href="https://github.com/ctrl-gaurav/effGen/blob/main/LICENSE"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                Read the licence
                <FiArrowRight size={14} />
              </a>
              <p className="mt-6 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Security reports go through{" "}
                <a
                  href="https://github.com/ctrl-gaurav/effGen/blob/main/SECURITY.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-700 dark:text-green-400 font-semibold"
                >
                  SECURITY.md
                </a>
                , not the public issue tracker.
              </p>
            </div>
          </motion.div>
        </Container>
      </section>

      {/* Join CTA */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="relative overflow-hidden p-10 lg:p-14 rounded-2xl bg-gray-50 dark:bg-gray-950/90 border border-gray-200 dark:border-green-500/20 shadow-sm dark:shadow-none text-center max-w-3xl mx-auto"
            style={{ boxShadow: "0 0 60px rgba(0,255,136,0.06)" }}
          >
            <motion.div
              className="absolute top-0 left-0 right-0 h-px"
              style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }}
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 3, repeat: Infinity }}
            />

            {/* Floating geometric shapes */}
            <motion.div
              className="absolute top-8 left-8 w-16 h-16 border border-green-500/10 rounded-lg pointer-events-none"
              animate={{ rotate: 360, scale: [1, 1.1, 1] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute bottom-8 right-8 w-12 h-12 border border-cyan-500/10 rounded-full pointer-events-none"
              animate={{ rotate: -360 }}
              transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute top-1/2 right-16 w-10 h-10 border border-violet-500/10 pointer-events-none"
              style={{ clipPath: "polygon(50% 0%, 100% 100%, 0% 100%)" }}
              animate={{ rotate: 360 }}
              transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            />

            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="inline-block mb-6 relative z-10"
            >
              <FiUsers className="mx-auto text-green-400" size={48} />
            </motion.div>
            <h2 className="text-3xl md:text-4xl font-black mb-4 text-gray-900 dark:text-white relative z-10">
              <span className="gradient-text">Come and say hello</span>
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto relative z-10">
              Questions go to Discord, bugs and feature requests to the issue tracker, and code to a
              pull request.
            </p>
            <div className="flex flex-wrap gap-4 justify-center relative z-10">
              <motion.a
                href="https://discord.com/invite/jacn9ed3"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)", boxShadow: "0 0 25px rgba(0,255,136,0.3)" }}
                whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,136,0.5)" }}
                whileTap={{ scale: 0.95 }}
              >
                <SiDiscord size={16} aria-hidden="true" />
                Join Discord
                <FiArrowRight size={14} />
              </motion.a>
              <motion.a
                href="https://github.com/ctrl-gaurav/effGen"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiGithub size={16} />
                Star on GitHub
              </motion.a>
            </div>
          </motion.div>
        </Container>
      </section>

      </main>
      <Footer />
    </div>
  );
}
