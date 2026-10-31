"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { siteData } from "./siteData";

// What the badge shows when PyPI cannot be reached — offline, blocked, or simply
// slow. It is the version `scripts/gen_site_data.py` read out of the installed
// framework, so the fallback is a derived number rather than one typed in that
// goes stale the next time the package is released.
const BUILT_VERSION = siteData.version;

interface PyPIVersionProps {
  className?: string;
  showLink?: boolean;
}

interface VersionInfo {
  version: string;
  loading: boolean;
  error: boolean;
}

export default function PyPIVersion({ className = "", showLink = true }: PyPIVersionProps) {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: BUILT_VERSION,
    loading: true,
    error: false,
  });

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const response = await fetch("https://pypi.org/pypi/effgen/json");

        if (!response.ok) {
          throw new Error("Failed to fetch PyPI data");
        }

        const data = await response.json();

        setVersionInfo({
          version: data.info?.version || BUILT_VERSION,
          loading: false,
          error: false,
        });
      } catch {
        // The badge is a live lookup over the built-in version, and the export is
        // required to render with the network off. A failed lookup is an expected
        // state, not an error: fall back to the version this site was built from.
        setVersionInfo({
          version: BUILT_VERSION,
          loading: false,
          error: true,
        });
      }
    };

    fetchVersion();
  }, []);

  if (versionInfo.loading) {
    return (
      <span className={`inline-block px-2 py-1 text-xs rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 animate-pulse ${className}`}>
        v...
      </span>
    );
  }

  const versionBadge = (
    <span className={`inline-block px-2 py-1 text-xs rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 font-semibold ${className}`}>
      v{versionInfo.version}
    </span>
  );

  if (showLink) {
    return (
      <motion.a
        href="https://pypi.org/project/effgen/"
        target="_blank"
        rel="noopener noreferrer"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {versionBadge}
      </motion.a>
    );
  }

  return versionBadge;
}

// Export a hook for use in other components
export function usePyPIVersion() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: BUILT_VERSION,
    loading: true,
    error: false,
  });

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const response = await fetch("https://pypi.org/pypi/effgen/json");

        if (!response.ok) {
          throw new Error("Failed to fetch PyPI data");
        }

        const data = await response.json();

        setVersionInfo({
          version: data.info?.version || BUILT_VERSION,
          loading: false,
          error: false,
        });
      } catch {
        setVersionInfo({
          version: BUILT_VERSION,
          loading: false,
          error: true,
        });
      }
    };

    fetchVersion();
  }, []);

  return versionInfo;
}
