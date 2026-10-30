import { useEffect, type RefObject } from 'react'

const FOCUSABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

/**
 * Keeps keyboard focus inside an open overlay, and hands it back when the
 * overlay closes.
 *
 * Without this, tabbing out of an open menu walks into the page behind it —
 * invisible to a pointer user and disorienting to anyone navigating by keyboard
 * or screen reader. Escape closes, Tab and Shift+Tab wrap at the ends, and
 * whatever was focused before the overlay opened gets focus back afterwards.
 *
 * This is the documentation site's copy of the behaviour the landing site's
 * mobile menu and modals use, so the two halves of the site answer a keyboard
 * the same way.
 *
 * @param ref       the overlay's container
 * @param active    whether the overlay is open
 * @param onClose   called on Escape
 */
export function useFocusTrap(
  ref: RefObject<HTMLElement | null>,
  active: boolean,
  onClose: () => void,
) {
  useEffect(() => {
    if (!active) return

    const container = ref.current
    const previouslyFocused = document.activeElement as HTMLElement | null

    const focusable = () =>
      Array.from(container?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []).filter(
        (el) => el.offsetParent !== null || el === document.activeElement,
      )

    // Move focus in, so the first Tab press lands inside rather than after.
    focusable()[0]?.focus()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }

      if (event.key !== 'Tab') return

      const items = focusable()
      if (items.length === 0) return

      const first = items[0]
      const last = items[items.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      previouslyFocused?.focus?.()
    }
  }, [ref, active, onClose])
}
