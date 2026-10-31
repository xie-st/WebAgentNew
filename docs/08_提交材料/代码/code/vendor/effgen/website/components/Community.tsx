"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useState, useRef, useCallback } from "react";
import { FiArrowRight, FiGithub, FiTwitter, FiUsers } from "react-icons/fi";
import { SiDiscord } from "react-icons/si";
import { FaLinkedin as SiLinkedin } from "react-icons/fa6";
import type { IconType } from 'react-icons';
import Link from "next/link";
import Container from "./Container";
import { useGitHubStats } from "./GitHubStats";
import { useReducedMotion } from "./useReducedMotion";
import { version } from "./siteData";
import { accentTextStyle } from "./accentText";

function formatNum(num: number): string {
  if (num >= 1000) return (num / 1000).toFixed(1) + "k";
  return num.toString();
}

// Brand colour glow mapping
const brandGlows: Record<string, string> = {
  GitHub: "rgba(232,234,237,0.4)",
  LinkedIn: "rgba(10,102,194,0.4)",
  "Twitter/X": "rgba(29,155,240,0.4)",
  Discord: "rgba(88,101,242,0.4)",
};

interface Place {
  icon: IconType;
  name: string;
  description: string;
  link: string;
  accent: string;
  /** Only shown where the number is one this site can actually read. */
  stats?: { label: string; value: string }[];
}

function CommunityCard({ place, index, inView }: { place: Place; index: number; inView: boolean }) {
  const cardRef = useRef<HTMLAnchorElement>(null);
  const reduced = useReducedMotion();
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  // The tilt is motion the visitor did not ask for, so it does not run when
  // they have asked for less of it. CSS cannot reach it — the rotation is an
  // inline transform written from the pointer position, not an animation.
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!cardRef.current || reduced) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    setTilt({
      x: ((y - centerY) / centerY) * -5,
      y: ((x - centerX) / centerX) * 5,
    });
  }, [reduced]);

  const handleMouseLeave = useCallback(() => {
    setTilt({ x: 0, y: 0 });
  }, []);

  return (
    <motion.a
      ref={cardRef}
      href={place.link}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ y: -8, scale: 1.02 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="group relative flex flex-col p-6 rounded-2xl bg-gray-50 dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden"
      style={{
        transform: `perspective(800px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
        transition: "transform 0.15s ease-out",
      }}
    >
      <motion.div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
        style={{ background: `radial-gradient(circle at 30% 30%, ${place.accent}10 0%, transparent 70%)` }}
      />
      <motion.div
        className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ boxShadow: `inset 0 0 0 1px ${place.accent}30` }}
      />

      <motion.div
        className="w-14 h-14 rounded-xl flex items-center justify-center mb-5 text-2xl relative z-10 transition-all duration-300"
        style={{
          background: `${place.accent}15`,
          border: `1px solid ${place.accent}25`,
        }}
        whileHover={{ rotate: 10, scale: 1.1, boxShadow: `0 0 20px ${brandGlows[place.name] || place.accent}` }}
      >
        <place.icon style={accentTextStyle(place.accent)} size={24} aria-hidden="true" />
      </motion.div>

      <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white relative z-10">{place.name}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-5 leading-relaxed relative z-10">
        {place.description}
      </p>

      {place.stats && (
        <div className="grid grid-cols-2 gap-3 mb-5 relative z-10">
          {place.stats.map((stat, idx) => (
            <div key={stat.label} className="text-center p-3 rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800">
              <motion.div
                className="text-xl font-black"
                style={accentTextStyle(place.accent)}
                initial={{ opacity: 0, y: 5 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.5 + index * 0.1 + idx * 0.1 }}
              >
                {stat.value}
              </motion.div>
              <div className="text-[10px] text-gray-600 dark:text-gray-400 font-medium uppercase tracking-wide">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      )}

      <motion.div
        className="mt-auto flex items-center gap-1.5 text-sm font-semibold relative z-10"
        style={accentTextStyle(place.accent)}
        whileHover={{ x: 4 }}
      >
        Open {place.name}
        <FiArrowRight size={14} />
      </motion.div>

      <div
        className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ background: `linear-gradient(90deg, transparent, ${place.accent}60, transparent)` }}
      />
    </motion.a>
  );
}

export default function Community() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const githubStats = useGitHubStats();

  // The GitHub card carries numbers because the GitHub API reports them and the
  // site reads them. The other three do not: a follower count nobody here can
  // read would be a number this page made up, so the card says what is there
  // instead of guessing how many people are.
  const places: Place[] = [
    {
      icon: FiGithub,
      name: "GitHub",
      description:
        "The source, the issue tracker and the releases. Every example on this site is a file in that repository.",
      stats: [
        { label: "Stars", value: githubStats.loading ? "…" : formatNum(githubStats.stars) },
        { label: "Contributors", value: githubStats.loading ? "…" : githubStats.contributors.toString() },
      ],
      link: "https://github.com/ctrl-gaurav/effGen",
      accent: "#e8eaed",
    },
    {
      icon: SiDiscord,
      name: "Discord",
      description:
        "Ask a question, say what you are building, or work through something that is not behaving.",
      link: "https://discord.com/invite/jacn9ed3",
      accent: "#5865f2",
    },
    {
      icon: FiTwitter,
      name: "Twitter/X",
      description: `Release notes and short posts about what changed — ${version} and whatever comes after it.`,
      link: "https://x.com/effGen_org",
      accent: "#1d9bf0",
    },
    {
      icon: SiLinkedin,
      name: "LinkedIn",
      description: "Longer write-ups, and where the project posts about the work behind a release.",
      link: "https://www.linkedin.com/company/111341317/",
      accent: "#0a66c2",
    },
  ];

  return (
    <section id="community" className="py-24 bg-white dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
            <FiUsers size={14} />
            Community
          </span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            Where the project <span className="gradient-text">is</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Open source, developed in the open, and reachable in four places.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {places.map((place, index) => (
            <CommunityCard key={place.name} place={place} index={index} inView={inView} />
          ))}
        </div>

        {/* CTA Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="relative overflow-hidden p-8 lg:p-12 rounded-2xl bg-gray-50 dark:bg-gray-900/80 border border-green-500/20 text-center"
          style={{ boxShadow: "0 0 60px rgba(0,255,136,0.05)" }}
        >
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <motion.div
            className="absolute top-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 3, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, #00c96e, transparent)" }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 3, repeat: Infinity, delay: 1.5 }}
          />

          <div className="relative z-10">
            <h3 className="text-2xl md:text-3xl font-black mb-3 text-gray-900 dark:text-white">
              Contributing, citing, or reporting something
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto">
              The community page has the contribution guide, the citation, the licence and
              the places to raise a problem.
            </p>
            <Link
              href="/community"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
              style={{
                background: "linear-gradient(135deg, #00ff88, #00c96e)",
                boxShadow: "0 0 30px rgba(0,255,136,0.3)",
              }}
            >
              Community page
              <FiArrowRight size={16} />
            </Link>
          </div>
        </motion.div>
      </Container>
    </section>
  );
}
