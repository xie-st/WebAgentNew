"use client";

import { useEffect, useState } from "react";

/**
 * Whether this device is likely to struggle with the full-motion background.
 *
 * The hero's backdrop animates a few dozen elements continuously. On a laptop
 * that is free; on a low-core phone it is the difference between a page that
 * scrolls and one that stutters. There is no web API that reports "this device
 * is on battery and thermally limited" — `navigator.getBattery` has been
 * removed from most browsers — so this uses the two capability signals that are
 * actually reported, and treats either one being small as a reason to draw
 * less:
 *
 *  - `hardwareConcurrency`, the number of logical cores.
 *  - `deviceMemory`, RAM in gigabytes, rounded down to a power of two. Not
 *    implemented in every browser; when it is absent it is simply not consulted.
 *
 * `data-power="low"` / `"full"` on `<html>` overrides both, the same way
 * `data-motion` overrides the reduced-motion query, so the reduced state can be
 * exercised in a review without finding a slower machine.
 *
 * It returns `false` on the server and on the first client render, which keeps
 * markup identical across hydration; the real value arrives in the effect
 * immediately afterwards.
 */
export function useLowPower(): boolean {
  const [lowPower, setLowPower] = useState(false);

  useEffect(() => {
    const resolve = () => {
      const override = document.documentElement.dataset.power;
      if (override === "low") return setLowPower(true);
      if (override === "full") return setLowPower(false);

      const nav = navigator as Navigator & { deviceMemory?: number };
      const cores = nav.hardwareConcurrency;
      const memory = nav.deviceMemory;

      setLowPower(
        (typeof cores === "number" && cores > 0 && cores <= 4) ||
          (typeof memory === "number" && memory > 0 && memory <= 4),
      );
    };

    resolve();

    const observer = new MutationObserver(resolve);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-power"],
    });

    return () => observer.disconnect();
  }, []);

  return lowPower;
}
