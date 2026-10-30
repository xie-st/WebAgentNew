import { useState, useEffect } from 'react'
import { siteData } from '../siteData'

interface VersionInfo {
  version: string
  loading: boolean
  error: boolean
}

// What the badge shows when PyPI cannot be reached — offline, blocked, or
// simply slow. It is the version `scripts/gen_site_data.py` read out of the
// installed framework, so the fallback is a derived number rather than one
// typed in that goes stale the next time the package is released.
const BUILT_VERSION = siteData.version

export function usePyPIVersion() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: BUILT_VERSION,
    loading: true,
    error: false,
  })

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const response = await fetch('https://pypi.org/pypi/effgen/json')

        if (!response.ok) {
          throw new Error('Failed to fetch PyPI data')
        }

        const data = await response.json()

        setVersionInfo({
          version: data.info?.version || BUILT_VERSION,
          loading: false,
          error: false,
        })
      } catch {
        setVersionInfo({
          version: BUILT_VERSION,
          loading: false,
          error: true,
        })
      }
    }

    fetchVersion()
  }, [])

  return versionInfo
}
