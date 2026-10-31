"use client";

import { ReactNode } from "react";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import Container from "@/components/Container";
import { accentTextStyle } from "@/components/accentText";

interface SectionProps {
  /** Anchor target, so a link can point straight at this band. */
  id?: string;
  /** The small pill above the heading — a category, not a sentence. */
  eyebrow?: string;
  /** Optional icon inside the pill. */
  eyebrowIcon?: ReactNode;
  /** Accent for the pill and its border. Defaults to the site's green. */
  accent?: string;
  /** The heading. Wrap a word in `<span className="gradient-text">` to accent it. */
  title: ReactNode;
  /** One sentence under the heading saying what the band is for. */
  lede?: ReactNode;
  /** `true` on a band that sits directly under another band of the same colour. */
  tinted?: boolean;
  className?: string;
  children: ReactNode;
}

/**
 * The section rhythm every band on the landing site and the product pages
 * shares: the vertical padding, the container, the centred header, and the
 * reveal that fires once the band scrolls into view.
 *
 * It exists so a new page does not re-derive spacing that `Features`,
 * `HowItWorks` and `QuickStart` already agreed on. The reveal goes through
 * framer-motion, which the root layout drives from the visitor's motion
 * preference — with motion off the band is simply present, in its settled
 * position.
 */
export default function Section({
  id,
  eyebrow,
  eyebrowIcon,
  accent = "#00ff88",
  title,
  lede,
  tinted = false,
  className = "",
  children,
}: SectionProps) {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.1 });

  return (
    <section
      id={id}
      ref={ref}
      className={`py-24 relative overflow-hidden ${
        tinted
          ? "bg-gray-50 dark:bg-[#04140c]"
          : "bg-white dark:bg-[#020c08]"
      } ${className}`}
    >
      <Container className="relative z-10">
        {(eyebrow || title || lede) && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            {eyebrow && (
              <span
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full border text-sm font-semibold mb-6"
                style={{
                  borderColor: `${accent}4d`,
                  backgroundColor: `${accent}0d`,
                  ...accentTextStyle(accent),
                }}
              >
                {eyebrowIcon}
                {eyebrow}
              </span>
            )}
            <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              {title}
            </h2>
            {lede && (
              <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
                {lede}
              </p>
            )}
          </motion.div>
        )}

        {children}
      </Container>
    </section>
  );
}
