"use client";

import { MotionConfig } from "framer-motion";
import { useEffect } from "react";
import { useReducedMotion } from "./useReducedMotion";

/**
 * Applies the visitor's motion preference to everything below it.
 *
 * Movement on this site comes from three places, and each needs its own answer:
 *
 *  - **CSS animations and transitions**, handled by the reduced-motion layer in
 *    `app/globals.css`, which the media query drives directly.
 *  - **framer-motion**, through `MotionConfig`. `reducedMotion: "always"` drops
 *    the positional values — width, height, the insets and the transforms — and
 *    `skipAnimations` keeps a new animation from starting at all.
 *  - **anything that asks `useReducedMotion()` itself**, such as a canvas loop
 *    that should paint one settled frame instead of running.
 *
 * `MotionConfig` alone is not enough, which is the reason for the effect below.
 * It governs animations that have yet to start; it does not reach the ones
 * already running, and it leaves an endless `opacity` or `backgroundColor`
 * pulse turning because neither is a positional value. Measured on the home
 * page with the preference set, forty-eight infinite animations were still
 * running with the config in place.
 *
 * So the running ones are settled directly, through the animations API the
 * browser exposes:
 *
 *  - an endless animation is **cancelled**, which returns its element to the
 *    style it has when nothing is animating — the resting composition the page
 *    was designed around, rather than a pulse stopped at an arbitrary frame;
 *  - a one-shot animation is **finished**, so a reveal lands on its end state
 *    and its content is visible rather than stuck at opacity zero.
 *
 * framer-motion does not restart what has been settled, so this runs when the
 * preference is set, when new elements arrive (a route change, a modal, an
 * accordion) and on scroll, which is when a `whileInView` reveal would begin.
 */
export default function MotionPreference({
  children,
}: {
  children: React.ReactNode;
}) {
  const reduced = useReducedMotion();

  useEffect(() => {
    if (!reduced) return;

    const settle = () => {
      for (const animation of document.getAnimations()) {
        const timing = animation.effect?.getTiming();
        if (!timing) continue;
        if (timing.iterations === Infinity) {
          animation.cancel();
        } else {
          try {
            animation.finish();
          } catch {
            // An animation with no resolved end time cannot be finished; it is
            // also not one that runs forever, so leaving it is safe.
          }
        }
      }
    };

    settle();

    // Only new elements are watched. Settling writes inline styles, so watching
    // attributes as well would have this observer answering its own edits.
    const observer = new MutationObserver(settle);
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("scroll", settle, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", settle);
    };
  }, [reduced]);

  return (
    <MotionConfig
      reducedMotion={reduced ? "always" : "user"}
      skipAnimations={reduced}
    >
      {children}
    </MotionConfig>
  );
}
