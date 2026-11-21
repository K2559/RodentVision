import { useLayoutEffect, useMemo, useRef, useState } from "react";

export type ReportRow = { id: string; clip: string; times: string };

function ReportTable({ rows }: { rows: ReportRow[] }) {
  return (
    <table aria-label="Littering Act Report">
      <thead><tr><th scope="col">Name of Video Clip</th><th scope="col">Time of the littering act / dog occurrence</th></tr></thead>
      <tbody>
        {rows.length ? rows.map(row => <tr key={row.id}><td>{row.clip}</td><td>{row.times}</td></tr>)
          : <tr><td colSpan={2}>Add video clips in Review.</td></tr>}
      </tbody>
    </table>
  );
}

export default function LitteringReport({ rows, fitPage }: { rows: ReportRow[]; fitPage: boolean }) {
  const measure = useRef<HTMLElement>(null);
  const stack = useRef<HTMLDivElement>(null);
  const [pageEnds, setPageEnds] = useState<number[]>([]);
  const [scale, setScale] = useState(1);
  // Split exceptionally long event lists into rows that can fit on a sheet.
  const printableRows = useMemo(() => rows.flatMap(row => {
    const times = row.times.split("\n");
    const parts: ReportRow[] = [];
    for (let start = 0; start < times.length; start += 16) {
      parts.push({
        id: row.id + ":" + start,
        clip: start ? row.clip + " (continued)" : row.clip,
        times: times.slice(start, start + 16).join("\n"),
      });
    }
    return parts;
  }), [rows]);

  useLayoutEffect(() => {
    const sheet = measure.current;
    if (!sheet) return;
    const body = sheet.querySelector<HTMLElement>(".report-sheet-body")!;
    const header = sheet.querySelector("thead")!.getBoundingClientRect().height;
    const available = body.getBoundingClientRect().height - header - 2;
    let used = 0;
    const ends: number[] = [];
    sheet.querySelectorAll("tbody tr").forEach((row, index) => {
      const height = row.getBoundingClientRect().height;
      if (used && used + height > available) {
        ends.push(index);
        used = 0;
      }
      used += height;
    });
    ends.push(printableRows.length);
    setPageEnds(ends);
  }, [printableRows]);

  useLayoutEffect(() => {
    const area = stack.current?.parentElement;
    const toolbar = area?.querySelector(".littering-report-toolbar");
    const sheet = measure.current;
    if (!fitPage || !area || !toolbar || !sheet) return;
    const fit = () => {
      const widthRatio = (area.clientWidth - 48) / sheet.offsetWidth;
      const heightRatio = (area.clientHeight - toolbar.getBoundingClientRect().height - 48) / sheet.offsetHeight;
      setScale(Math.max(.25, Math.min(1, widthRatio, heightRatio)));
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(area);
    observer.observe(toolbar);
    return () => observer.disconnect();
  }, [fitPage]);

  const ends = pageEnds.length ? pageEnds : [printableRows.length];
  return (
    <div className="report-paper-stack" ref={stack}>
      <article ref={measure} className="report-paper report-measure" aria-hidden="true">
        <h1>Littering Act Report</h1>
        <div className="report-sheet-body"><ReportTable rows={printableRows} /></div>
        <footer>Page 1 of 1</footer>
      </article>
      {ends.map((end, index) => (
        <article className="report-paper" style={{ zoom: fitPage ? scale : 1 }} key={index} aria-label={"A4 page " + (index + 1)}>
          <h1>Littering Act Report</h1>
          <div className="report-sheet-body"><ReportTable rows={printableRows.slice(ends[index - 1] ?? 0, end)} /></div>
          <footer>Page {index + 1} of {ends.length}</footer>
        </article>
      ))}
    </div>
  );
}
