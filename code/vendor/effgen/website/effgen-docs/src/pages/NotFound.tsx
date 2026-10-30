import { Link } from 'react-router-dom'
import DocPage from '../components/DocPage'

// Where someone who hit a dead documentation URL most likely meant to go. A 404
// that only says "not found" sends the reader back to the sidebar to start
// again; these four cover almost every wrong address on this site.
const destinations = [
  {
    path: '/introduction',
    title: 'Introduction',
    blurb: 'What effGen is, and what an agent run actually does.',
  },
  {
    path: '/quickstart',
    title: 'Quick start',
    blurb: 'Install it and get an agent answering in a couple of minutes.',
  },
  {
    path: '/api-reference',
    title: 'API reference',
    blurb: 'Every public name, with its signature and what it returns.',
  },
  {
    path: '/examples',
    title: 'Examples',
    blurb: 'Complete programs with the output they actually produce.',
  },
]

export default function NotFound() {
  return (
    <DocPage
      title="Page not found"
      subtitle="That documentation page does not exist. The link may be out of date, or the address may have a typo in it."
      breadcrumbs={[{ label: '404' }]}
    >
      <p>
        Some pages moved when the documentation was rebuilt for 1.0.0. If you
        followed a link from elsewhere on this site, the four below cover most of
        what people are looking for; the sidebar carries the full set.
      </p>

      <div className="quick-links">
        {destinations.map((destination) => (
          <Link key={destination.path} to={destination.path} className="quick-link-card">
            <div className="quick-link-title">{destination.title}</div>
            <div className="quick-link-desc">{destination.blurb}</div>
          </Link>
        ))}
      </div>

      <p>
        If a link on this site brought you here,{' '}
        <a
          href="https://github.com/ctrl-gaurav/effGen/issues"
          target="_blank"
          rel="noopener noreferrer"
        >
          tell us where it was
        </a>
        .
      </p>
    </DocPage>
  )
}
