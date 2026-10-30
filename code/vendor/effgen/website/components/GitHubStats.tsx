"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FiStar, FiGitBranch, FiUsers } from "react-icons/fi";
import githubData from "@/data/github.json";

interface GitHubStatsProps {
  className?: string;
  showForks?: boolean;
  showContributors?: boolean;
}

interface Stats {
  stars: number;
  forks: number;
  contributors: number;
  loading: boolean;
  error: boolean;
}

const REPO = "ctrl-gaurav/effGen";
const API = `https://api.github.com/repos/${REPO}`;

// The figures the site ships, written by scripts/gen_github_data.py. They are
// what renders before the network answers and what stays on the page if it never
// does. The site used to start these at zero and only replace them once the API
// came back, so a visitor over the rate limit read "0 stars" — and the hero's row
// removed itself entirely.
const SEED: Stats = {
  stars: githubData.stars,
  forks: githubData.forks,
  contributors: githubData.contributors,
  loading: false,
  error: false,
};

// The unauthenticated GitHub API allows 60 requests an hour per address, and the
// home page has three of these on it. Fetching per component cost six requests a
// view — ten views an hour and every visitor after that saw zeros. One request
// pair per page load is shared by every caller through this promise, and the
// result is cached for the session, so a second page costs nothing.
const CACHE_KEY = "effgen_github_stats";
const CACHE_TTL_MS = 30 * 60 * 1000;

let inFlight: Promise<Stats> | null = null;

function readCache(): Stats | null {
  try {
    const raw = window.sessionStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const { at, stats } = JSON.parse(raw) as { at: number; stats: Stats };
    if (Date.now() - at > CACHE_TTL_MS) return null;
    return stats;
  } catch {
    return null;
  }
}

function writeCache(stats: Stats) {
  try {
    window.sessionStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ at: Date.now(), stats }),
    );
  } catch {
    /* a full or disabled store is not worth failing a render over */
  }
}

async function fetchStats(): Promise<Stats> {
  const cached = readCache();
  if (cached) return cached;

  try {
    const repoResponse = await fetch(API);
    if (!repoResponse.ok) throw new Error(`repo lookup returned ${repoResponse.status}`);
    const repo = await repoResponse.json();

    // Asking for one contributor per page makes the last page number the total.
    let contributors = SEED.contributors;
    try {
      const response = await fetch(`${API}/contributors?per_page=1&anon=false`);
      if (response.ok) {
        const link = response.headers.get("Link");
        const match = link?.match(/[?&]page=(\d+)>;\s*rel="last"/);
        if (match) {
          contributors = parseInt(match[1], 10);
        } else {
          const people = await response.json();
          if (Array.isArray(people) && people.length > 0) contributors = people.length;
        }
      }
    } catch {
      /* keep the shipped figure */
    }

    const stats: Stats = {
      // A live zero is still a zero, but an absent field is not — fall back to
      // the shipped figure only when the API did not report one.
      stars: typeof repo.stargazers_count === "number" ? repo.stargazers_count : SEED.stars,
      forks: typeof repo.forks_count === "number" ? repo.forks_count : SEED.forks,
      contributors,
      loading: false,
      error: false,
    };
    writeCache(stats);
    return stats;
  } catch {
    // Rate-limited, offline, or blocked. Show what the site shipped and say so
    // internally; the reader sees numbers rather than zeros or a missing row.
    return { ...SEED, error: true };
  }
}

function useStats(): Stats {
  const [stats, setStats] = useState<Stats>(SEED);

  useEffect(() => {
    let alive = true;
    if (!inFlight) inFlight = fetchStats();
    inFlight.then((resolved) => {
      if (alive) setStats(resolved);
    });
    return () => {
      alive = false;
    };
  }, []);

  return stats;
}

export default function GitHubStats({
  className = "",
  showForks = true,
  showContributors = true,
}: GitHubStatsProps) {
  const stats = useStats();

  const formatNumber = (num: number): string =>
    num >= 1000 ? (num / 1000).toFixed(1) + "k" : num.toString();

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Stars */}
      <motion.a
        href="https://github.com/ctrl-gaurav/effGen"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-yellow-400/10 text-yellow-600 dark:text-yellow-400 hover:bg-yellow-400/20 border border-yellow-400/20 hover:border-yellow-400/40 transition-all text-xs font-semibold font-mono"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FiStar size={12} />
        {formatNumber(stats.stars)} stars
      </motion.a>

      {/* Forks */}
      {showForks && (
        <motion.a
          href="https://github.com/ctrl-gaurav/effGen/fork"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-green-400/10 text-green-700 dark:text-green-400 hover:bg-green-400/20 border border-green-400/20 hover:border-green-400/40 transition-all text-xs font-semibold font-mono"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiGitBranch size={12} />
          {formatNumber(stats.forks)} forks
        </motion.a>
      )}

      {/* Contributors */}
      {showContributors && stats.contributors > 0 && (
        <motion.a
          href="https://github.com/ctrl-gaurav/effGen/graphs/contributors"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-cyan-400/10 text-cyan-600 dark:text-cyan-400 hover:bg-cyan-400/20 border border-cyan-400/20 hover:border-cyan-400/40 transition-all text-xs font-semibold font-mono"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiUsers size={12} />
          {stats.contributors} contributors
        </motion.a>
      )}
    </div>
  );
}

// The same figures, for the sections that lay them out themselves.
export function useGitHubStats(): Stats {
  return useStats();
}
