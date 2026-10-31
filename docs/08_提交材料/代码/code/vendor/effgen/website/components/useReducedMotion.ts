"use client";

import { useSyncExternalStore } from "react";

/**
 * Whether this visitor has asked for reduced motion.
 *
 * Two signals feed it, in this order:
 *
 *  1. `data-motion` on `<html>` — `"reduced"` forces reduction on and `"full"`
 *     forces it off. Nothing sets this in normal use; it exists so the reduced
 *     state can be exercised in a test or a review without changing an OS
 *     setting, which is otherwise the only way to reach it.
 *  2. the `prefers-reduced-motion: reduce` media query.
 *
 * Both are watched, so a change to either takes effect without a reload.
 *
 * **Why `useSyncExternalStore` and not `useState` + `useEffect`.** The obvious
 * shape — start at `false`, correct it in an effect — is what this hook used to
 * do, and it let every `repeat: Infinity` animation on the page start before the
 * preference arrived. Effects run child-first, so the animations were already
 * running by the time the provider above them learned to stop them, and an
 * animation that has begun is not covered by `skipAnimations`. This hook hands
 * React a server snapshot of `false` for hydration and the real value for the
 * first client render, so the answer is right before anything starts moving.
 */

function subscribe(onChange: () => void): () => void {
  const query = window.matchMedia("(prefers-reduced-motion: reduce)");
  query.addEventListener("change", onChange);

  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-motion"],
  });

  return () => {
    query.removeEventListener("change", onChange);
    observer.disconnect();
  };
}

function getSnapshot(): boolean {
  const override = document.documentElement.dataset.motion;
  if (override === "reduced") return true;
  if (override === "full") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// The prerender has no window to ask, and the markup it produces is the
// full-motion one; the client corrects it on its first render.
function getServerSnapshot(): boolean {
  return false;
}

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
