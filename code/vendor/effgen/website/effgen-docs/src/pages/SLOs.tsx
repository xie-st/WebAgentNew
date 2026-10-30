import { Target } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function SLOs() {
  return (
    <DocPage
      subtitle="Error-budget tracking over a rolling window, and the alert rules that go with it."
      icon={<Target size={48} />}
    >
      <p>
        An SLO here is a named objective — "99% of model calls succeed over an hour" — with a rolling
        window of good and bad events behind it and a burn rate computed from them. The tracker runs
        in process, a running <Link to="/api-server">server</Link> exposes it at{' '}
        <code>GET /slo</code>, and <code>check_slo_and_alert</code> turns a burn rate over threshold
        into a page.
      </p>

      <h2>Track one</h2>

      <CodeBlock filename="first.py" code={`from effgen.observability.slo import SLO, SLOTracker

tracker = SLOTracker()
tracker.register(SLO(name="model_call_success", target_pct=99.0, window_seconds=3600))

for _ in range(98):
    tracker.record("model_call_success", ok=True)
for _ in range(2):
    tracker.record("model_call_success", ok=False)

print("burn rate:", tracker.burn_rate("model_call_success"))
print(tracker.status("model_call_success"))`} />

      <Terminal
        command="python first.py"
        output={`burn rate: 2.0
{'name': 'model_call_success', 'target_pct': 99.0, 'window_seconds': 3600, 'query': '', 'total_events': 100, 'good_events': 98, 'bad_events': 2, 'good_ratio': 0.98, 'bad_ratio': 0.02, 'burn_rate': 2.0, 'within_budget': False}`}
        caption="Two bad events in a hundred against a 99% target is twice the budget — a burn rate of 2."
      />

      <h2>The arithmetic</h2>

      <ApiTable
        headers={['Term', 'What it is']}
        rows={[
          ['SLO', 'A name, a target percentage, and a window in seconds.'],
          [
            'Error budget',
            <>
              The fraction of events allowed to fail: <code>(100 − target_pct) / 100</code>. A 99%
              target has a budget of <code>0.01</code>.
            </>,
          ],
          [
            'Burn rate',
            <>
              <code>bad_ratio / error_budget_fraction</code>. <code>1.0</code> is exactly on budget;
              above that is over.
            </>,
          ],
          [
            'Rolling window',
            <>
              Only events inside the last <code>window_seconds</code> count. Old ones are evicted
              lazily on the next read, so nothing runs in the background.
            </>,
          ],
        ]}
      />

      <CodeBlock filename="budget.py" code={`from effgen.observability.slo import SLO

slo = SLO(name="model_call_success", target_pct=99.0, window_seconds=3600)
print("error budget fraction:", slo.error_budget_fraction)

for bad in (0, 1, 2, 14):
    tracker_state = bad / 100
    print(f"{bad:3} bad of 100 -> bad_ratio {tracker_state:.2f}  burn {tracker_state / slo.error_budget_fraction:.1f}x")`} />

      <Terminal command="python budget.py" output={`error budget fraction: 0.01
  0 bad of 100 -> bad_ratio 0.00  burn 0.0x
  1 bad of 100 -> bad_ratio 0.01  burn 1.0x
  2 bad of 100 -> bad_ratio 0.02  burn 2.0x
 14 bad of 100 -> bad_ratio 0.14  burn 14.0x`} />

      <Callout type="note" title="The two edge cases">
        <p>
          <code>burn_rate()</code> returns <code>0.0</code> when there are no bad events at all, and{' '}
          <code>float("inf")</code> for a 100%-target SLO that has any — because a budget of zero
          cannot be divided into. Neither is an error; both are what the number means.
        </p>
      </Callout>

      <h2>The API</h2>

      <ParamTable
        nameLabel="SLO field"
        params={[
          { name: 'name', type: 'str', required: true, description: 'Unique identifier, and what you record against.' },
          { name: 'target_pct', type: 'float', required: true, description: 'The success target, as a percentage.' },
          { name: 'window_seconds', type: 'int', required: true, description: 'The rolling window.' },
          {
            name: 'query',
            type: 'str',
            default: "''",
            description: 'An informal label saying what "good" meant. Recorded, never evaluated.',
          },
        ]}
        caption={
          <>
            <code>slo.error_budget_fraction</code> is computed from <code>target_pct</code>.
          </>
        }
      />

      <ApiTable
        headers={['SLOTracker method', 'What it does']}
        rows={[
          [
            <code>register(slo)</code>,
            'Register an objective. Idempotent, so calling it at import time on every startup is safe.',
          ],
          [
            <code>record(name, *, ok, ts=None)</code>,
            <>
              One event. <code>ts</code> is for replay and tests; the default is{' '}
              <code>time.monotonic()</code>.
            </>,
          ],
          [<code>burn_rate(name)</code>, 'The current burn rate over the window.'],
          [
            <code>status(name)</code>,
            'A JSON-serialisable dict: totals, good and bad counts and ratios, the burn rate, and whether it is within budget.',
          ],
          [<code>all_statuses()</code>, 'The same for every registered SLO, sorted by name.'],
          [<code>list_slos()</code>, 'The registered names.'],
        ]}
        caption={
          <>
            <code>get_tracker()</code> — also exported as <code>get_slo_tracker</code> — returns the
            process-wide tracker, which is the one <code>/slo</code> reports on.
          </>
        }
      />

      <CodeBlock filename="multiple.py" code={`from effgen.observability import get_slo_tracker
from effgen.observability.slo import SLO

tracker = get_slo_tracker()
tracker.register(SLO("model_call_success", 99.0, 3600))
tracker.register(SLO("tool_call_success", 99.5, 3600))
tracker.register(SLO("agent_run_success", 99.0, 86400))

print(tracker.list_slos())
for status in tracker.all_statuses():
    print(status["name"], status["target_pct"], status["within_budget"])`} />

      <Terminal command="python multiple.py" output={`['agent_run_success', 'model_call_success', 'tool_call_success']
agent_run_success 99.0 True
model_call_success 99.0 True
tool_call_success 99.5 True`} />

      <h2>Over HTTP</h2>

      <Terminal command="curl -s http://127.0.0.1:8000/slo" output={`{"slos":[],"detail":"No SLO objectives are registered in this process. This endpoint reports registered objectives only; measured latency percentiles and availability for served traffic are in the 'slo' block of /dashboard/data.json."}`} title="curl" />

      <Callout type="warning" title="An empty /slo is not a broken /slo">
        <p>
          <code>/slo</code> reports the objectives <em>this server process registered</em>. It does
          not invent one from request metrics, so a server that has served plenty of traffic and
          registered nothing returns an empty list with a <code>detail</code> note saying so.
          Measured latency percentiles, error-rate burn and availability for traffic actually served
          live in the <code>slo</code> block of <code>GET /dashboard/data.json</code>, and are what{' '}
          <Link to="/dashboard">the dashboard</Link> and <code>effgen top</code> draw.
        </p>
      </Callout>

      <p>
        The endpoint is public — it carries no request bodies, costs or user data — so an external
        poller can read burn rates without a credential.
      </p>

      <h2>Alert thresholds</h2>

      <ApiTable
        headers={['Window', 'Burn rate over', 'What it means']}
        rows={[
          ['1 hour', '14.4×', '5% of a monthly error budget spent in an hour. Page someone.'],
          ['6 hours', '6×', '5% spent in six hours.'],
          ['3 days', '1×', 'The budget runs out before the month does.'],
        ]}
        caption="The Google SRE fast-burn and slow-burn pattern, for a 99% objective. These are the numbers the bundled rule pack uses."
      />

      <h2>Firing an alert</h2>

      <p>
        <code>AlertWebhook</code> posts an <code>Alert</code> to a Slack or Discord webhook — the URL
        type is detected — or to any other receiver as an Alertmanager-style JSON POST.
      </p>

      <CodeBlock filename="alert.py" code={`import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from effgen.observability.alerting import Alert, AlertSeverity, AlertWebhook


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        print("receiver got:", json.dumps(json.loads(body), sort_keys=True)[:160])
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 9099), Receiver)
threading.Thread(target=server.serve_forever, daemon=True).start()

hook = AlertWebhook("http://127.0.0.1:9099/alerts")
print(hook.fire(Alert(
    name="HighErrorRate",
    severity=AlertSeverity.CRITICAL,
    summary="Error rate exceeded 5% for 10 minutes",
    value=0.08,
    threshold=0.05,
    labels={"provider": "openai", "model": "gpt-5-nano"},
)))
server.shutdown()`} />

      <Terminal
        command="python alert.py"
        output={`receiver got: {"alerts": [{"description": "", "fired_at": 1787542970.1978683, "labels": {"model": "gpt-5-nano", "provider": "openai"}, "name": "HighErrorRate", "severity": "c
{'ok': True, 'webhook': 'http://127.0.0.1:9099/***', 'status': 200, 'body': ''}`}
        caption="Captured against a local receiver, so the run is reproducible and nothing leaves the machine. Point it at a Slack or Discord URL and the payload becomes that service's message format instead."
      />

      <ParamTable
        nameLabel="Alert field"
        params={[
          { name: 'name', type: 'str', required: true, description: 'The alert name, as it appears in the message.' },
          {
            name: 'severity',
            type: 'AlertSeverity',
            required: true,
            description: (
              <>
                <code>INFO</code>, <code>WARNING</code> or <code>CRITICAL</code>.
              </>
            ),
          },
          { name: 'summary', type: 'str', required: true, description: 'One line saying what happened.' },
          { name: 'value', type: 'float', description: 'The observed value.' },
          { name: 'threshold', type: 'float', description: 'What it crossed.' },
          { name: 'labels', type: 'dict[str, str]', description: 'Extra dimensions — provider, model, anything you route on.' },
        ]}
      />

      <h3>It never raises, and it never logs the URL</h3>

      <p>
        A webhook URL carries its secret in the path, so effGen logs only the origin. And{' '}
        <code>fire()</code> catches everything: a network error, an HTTP error, a malformed payload
        all come back as <code>{'{"ok": False, "error": …}'}</code>. Alert delivery failing is not
        allowed to fail the agent loop that noticed the problem.
      </p>

      <CodeBlock filename="redaction.py" code={`from effgen.observability.alerting import Alert, AlertSeverity, AlertWebhook

# A host that cannot resolve, so nothing is sent — the point is the return value.
hook = AlertWebhook("https://hooks.slack.com.invalid/services/T00000/B00000/SECRETTOKEN")
result = hook.fire(Alert(name="Probe", severity=AlertSeverity.WARNING, summary="probe"))
print(result["ok"])
print(result["webhook"])`} />

      <Terminal command="python redaction.py" output={`False
https://hooks.slack.com.invalid/***`} />

      <h3>The scheduled check</h3>

      <p>
        <code>check_slo_and_alert</code> is the "read the meter, decide, fire" loop as one call. It
        returns the <code>fire()</code> result when the SLO is over threshold and <code>None</code>{' '}
        when it is within budget — so calling it on a schedule for each objective is the whole
        integration.
      </p>

      <CodeBlock filename="check.py" code={`import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from effgen.observability.alerting import AlertWebhook, check_slo_and_alert
from effgen.observability.slo import SLO, SLOTracker


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        print("paged:", body["summary"])
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 9098), Receiver)
threading.Thread(target=server.serve_forever, daemon=True).start()

tracker = SLOTracker()
tracker.register(SLO("model_call_success", 99.0, 3600))
for _ in range(80):
    tracker.record("model_call_success", ok=True)
for _ in range(20):
    tracker.record("model_call_success", ok=False)

webhook = AlertWebhook("http://127.0.0.1:9098/alerts")
print("burn rate:", round(tracker.burn_rate("model_call_success"), 1))
print("fired:", check_slo_and_alert(tracker, "model_call_success", webhook, burn_rate_threshold=14.4) is not None)

healthy = SLOTracker()
healthy.register(SLO("tool_call_success", 99.0, 3600))
for _ in range(100):
    healthy.record("tool_call_success", ok=True)
print("within budget ->", check_slo_and_alert(healthy, "tool_call_success", webhook))
server.shutdown()`} />

      <Terminal command="python check.py" output={`burn rate: 20.0
fired: True
within budget -> None`} />

      <h2>The Prometheus rule pack</h2>

      <p>
        <code>docs/observability/alert_rules.yaml</code> in the framework repository defines six
        alerts across five groups. Load it into Prometheus with a <code>rule_files</code> entry, or
        take it as a starting point for your own.
      </p>

      <ApiTable
        headers={['Alert', 'Severity', 'Condition', 'For']}
        rows={[
          [<code>HighErrorRate</code>, 'critical', 'Error rate above 5%', '10 min'],
          [<code>HighP95Latency</code>, 'warning', 'p95 latency above 10s', '5 min'],
          [<code>CostBurnHigh</code>, 'warning', 'Estimated cost above $10/day', 'instant'],
          [<code>SLOFastBurn</code>, 'critical', 'Burn rate above 14.4×', 'instant'],
          [<code>SLOSlowBurn</code>, 'warning', 'Burn rate above 3×', '60 min'],
          [<code>CircuitBreakerOpen</code>, 'warning', 'A breaker open', '1 min'],
        ]}
        caption={
          <>
            The queries behind them read the instruments on{' '}
            <Link to="/metrics">the metrics page</Link>.
          </>
        }
      />

      <CodeBlock filename="validate.py" code={`from pathlib import Path

from effgen.observability.alerting import validate_alert_rules_yaml

rules = Path("alert_rules.yaml")
ok, errors = validate_alert_rules_yaml(rules)
print("valid:", ok, errors)`} />

      <Terminal
        command="python validate.py"
        output={`valid: True []`}
        caption={
          <>
            The rule file is a repository artefact rather than something{' '}
            <code>pip install effgen</code> puts on disk — this ran against a copy taken from the
            checkout. <code>promtool check rules</code> validates the same file if you have it.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>/slo</code> returns an empty list
            </>,
            'No objective was registered in that process.',
            <>
              Register one at startup. The endpoint reports what you declared, not what it can infer
              — the dashboard's <code>slo</code> block is the inferred view.
            </>,
          ],
          [
            <>
              <code>burn_rate()</code> is <code>inf</code>
            </>,
            <>
              <code>target_pct=100</code>, so the budget is zero and any failure is infinite burn.
            </>,
            'Use 99.9 or 99.99. A 100% objective has no error budget to burn, by definition.',
          ],
          [
            'The burn rate drops on its own without anything being fixed',
            'The window rolled forward and the bad events aged out.',
            'Expected. Alert on the burn rate crossing a threshold for a duration, not on one reading.',
          ],
          [
            <>
              <code>fire()</code> returns <code>ok: False</code> and nothing is paged
            </>,
            'Delivery failed — a bad URL, a network problem, a receiver that rejected the payload.',
            <>
              Read <code>result["error"]</code>. It never raises, so a failed alert is silent unless
              you check the return value.
            </>,
          ],
          [
            'Alerts fire on every check',
            <>
              <code>burn_rate_threshold</code> defaults to <code>1.0</code> — exactly on budget.
            </>,
            'Pass 14.4 for a fast-burn page, or 3 for the slow-burn warning.',
          ],
          [
            'A separate process shows different numbers',
            'The tracker is per process; each worker has its own window.',
            <>
              Aggregate in Prometheus rather than in the tracker —{' '}
              <Link to="/metrics">Metrics</Link> is the cross-process view.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/metrics', '/reliability', '/observability']} />
    </DocPage>
  );
}
