import { useEffect, useState } from 'react'

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
 * Both are watched, so a change to either takes effect without a reload. This
 * is the same contract as the landing site's hook of the same name.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')

    const resolve = () => {
      const override = document.documentElement.dataset.motion
      if (override === 'reduced') return setReduced(true)
      if (override === 'full') return setReduced(false)
      setReduced(query.matches)
    }

    resolve()
    query.addEventListener('change', resolve)

    const observer = new MutationObserver(resolve)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-motion'],
    })

    return () => {
      query.removeEventListener('change', resolve)
      observer.disconnect()
    }
  }, [])

  return reduced
}
