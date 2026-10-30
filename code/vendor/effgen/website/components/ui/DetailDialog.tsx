"use client";

import { motion } from "framer-motion";
import { useId, useRef, type ReactNode } from "react";
import { FiX } from "react-icons/fi";
import { useFocusTrap } from "@/components/useFocusTrap";

interface DetailDialogProps {
  /** Heading, and the accessible name of the dialog. */
  title: string;
  /** Colour for the header hairline and the panel glow. */
  accent?: string;
  onClose: () => void;
  children: ReactNode;
}

/**
 * The panel a condensed card opens.
 *
 * The landing page shows a capability as a short card and keeps the detail
 * behind it, so several sections need the same overlay. Each one that wrote its
 * own repeated the same twenty lines of keyboard handling; this uses
 * `useFocusTrap`, which is the copy that also moves focus into the panel when it
 * opens rather than leaving it on the card underneath.
 *
 * Escape closes, Tab stays inside, and the element that opened the dialog is
 * focused again when it closes.
 */
export default function DetailDialog({
  title,
  accent = "#00ff88",
  onClose,
  children,
}: DetailDialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useFocusTrap(panelRef, true, onClose);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      />

      <motion.div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl"
        style={{ boxShadow: `0 0 50px ${accent}15` }}
        initial={{ scale: 0.94, opacity: 0, y: 24 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.94, opacity: 0, y: 24 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        <div
          className="h-1 rounded-t-2xl"
          style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
        />

        <div className="p-6">
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute top-4 right-4 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
          >
            <FiX size={16} />
          </button>

          <h3
            id={titleId}
            className="text-xl font-black text-gray-900 dark:text-white mb-4 pr-10"
          >
            {title}
          </h3>

          {children}
        </div>
      </motion.div>
    </motion.div>
  );
}
