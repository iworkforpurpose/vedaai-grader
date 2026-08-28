"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import { useNarrow } from "@/lib/breakpoints";
import {
  AnalyticsIcon,
  ChevronsRightIcon,
  AssignmentsIcon,
  BackIcon,
  BellIcon,
  ChevronDownIcon,
  ClassroomIcon,
  ExamsIcon,
  HelpIcon,
  HomeIcon,
  LibraryIcon,
  MenuIcon,
  PanelIcon,
  ReviewIcon,
  SettingsIcon,
  SparkleIcon,
} from "./icons";

/**
 * The application frame: navigation rail, top bar, and a slot for the screen.
 *
 * One set of markup for both layouts. On the phone the rail is a drawer over the
 * content; from 1024px it is a column beside it. Keeping a second copy of the
 * navigation for mobile is how the two silently drift apart, so the difference
 * lives entirely in CSS.
 *
 * Only Exams is reachable, as scoped. The rest is rendered exactly as the file
 * shows it — present and legible, plainly unavailable — because the point is that
 * the wider product exists, which hiding the items would misrepresent. Each is
 * inert three independent ways: no href, `aria-disabled` so assistive technology
 * announces it, and `pointer-events: none` so a click cannot land even if the
 * styling is overridden.
 *
 * The item list comes from the API rather than the PNG. The supplied export shows
 * five entries; the file has seven, and My Library carries a badge. A stale
 * screenshot is exactly the kind of thing that makes a "faithful" rebuild wrong.
 */

type NavEntry = {
  label: string;
  icon: React.JSX.Element;
  badge?: string;
  available?: boolean;
};

const NAV: NavEntry[] = [
  { label: "Home", icon: <HomeIcon /> },
  { label: "My Classroom", icon: <ClassroomIcon /> },
  { label: "Assignments", icon: <AssignmentsIcon /> },
  { label: "Exams", icon: <ExamsIcon />, available: true },
  { label: "My Library", icon: <LibraryIcon />, badge: "32" },
  { label: "Review", icon: <ReviewIcon /> },
  { label: "Analytics", icon: <AnalyticsIcon /> },
];

export function AppShell({
  crumb,
  onBack,
  collapsedRail = false,
  children,
}: {
  crumb: string;
  onBack?: () => void;
  /**
   * Start with the rail collapsed to icons.
   *
   * The loading and mapping frames show it that way and the upload frame does not,
   * which reads as intent rather than inconsistency: those screens want the width
   * for content. It stays a state the teacher can change either way.
   */
  collapsedRail?: boolean;
  children: React.ReactNode;
}): React.JSX.Element {
  const [navOpen, setNavOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(collapsedRail);
  const narrow = useNarrow();

  // Follow the prop when it changes, which is the upload-to-loading transition.
  // `useState` takes it as an initial value only, so without this the rail stayed
  // expanded through a screen the frame draws collapsed. A manual toggle in
  // between survives, because this runs only when the prop itself changes.
  useEffect(() => {
    setCollapsed(collapsedRail);
  }, [collapsedRail]);

  // Escape closes the drawer — expected of anything that covers the page, and
  // cheap enough that omitting it is just an omission.
  useEffect(() => {
    if (!navOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNavOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navOpen]);

  return (
    <div className="shell">
      {navOpen && (
        <button
          type="button"
          className="rail-scrim"
          aria-label="Close navigation"
          onClick={() => setNavOpen(false)}
        />
      )}

      <nav className="rail" data-open={navOpen} data-collapsed={collapsed} aria-label="Main">
        <div className="rail-head">
          <span className="brand-mark">
            <Image src="/brand/logo.png" alt="" width={40} height={40} priority />
          </span>
          <span className="brand-word">VedaAI</span>
          <button
            type="button"
            className="rail-collapse"
            /*
             * The same control, two jobs. Below the rail breakpoint there is
             * nothing to collapse to — the rail is a drawer — so its only effect
             * is closing it, and it is the only visible way to do that besides
             * the scrim. Announcing "collapse" there described something that
             * does not happen.
             */
            aria-label={narrow ? "Close navigation" : "Collapse navigation"}
            onClick={() => {
              setNavOpen(false);
              setCollapsed(true);
            }}
          >
            <PanelIcon />
          </button>
        </div>

        <button type="button" className="toolkit" title="AI Teacher's Toolkit">
          <SparkleIcon size={17} />
          <span className="nav-label">AI Teacher&rsquo;s Toolkit</span>
        </button>

        <ul className="rail-menu">
          {NAV.map((entry) => (
            <li key={entry.label}>
              {entry.available ? (
                <a className="nav-row" href="/" aria-current="page" title={entry.label}>
                  <span className="nav-icon">{entry.icon}</span>
                  <span className="nav-label">{entry.label}</span>
                </a>
              ) : (
                <span className="nav-row" aria-disabled="true" title={entry.label}>
                  <span className="nav-icon">{entry.icon}</span>
                  <span className="nav-label">{entry.label}</span>
                  {entry.badge && <span className="nav-badge">{entry.badge}</span>}
                </span>
              )}
            </li>
          ))}
        </ul>

        <div className="rail-foot">
          <span className="nav-row" aria-disabled="true" title="Settings">
            <span className="nav-icon">
              <SettingsIcon />
            </span>
            <span className="nav-label">Settings</span>
          </span>

          <div className="school">
            <span className="school-crest">
              <Image src="/brand/school-crest.png" alt="" width={59} height={60} />
            </span>
            <span className="school-text">
              <span className="school-name">Delhi Public School</span>
              <br />
              <span className="school-place">Bokaro Steel City</span>
            </span>
          </div>

          {collapsed && (
            <button
              type="button"
              className="rail-expand"
              aria-label="Expand navigation"
              onClick={() => setCollapsed(false)}
            >
              <ChevronsRightIcon />
            </button>
          )}
        </div>
      </nav>

      <div className="shell-main">
        <header className="topbar">
          <button
            type="button"
            className="round-button only-wide"
            aria-label="Back"
            onClick={onBack}
          >
            <BackIcon size={22} />
          </button>
          <button
            type="button"
            className="round-button only-narrow"
            aria-label="Open navigation"
            aria-expanded={navOpen}
            onClick={() => setNavOpen(true)}
          >
            <MenuIcon />
          </button>

          <span className="crumb">
            <span className="nav-icon only-wide">
              <ExamsIcon size={20} />
            </span>
            <span>{crumb}</span>
          </span>

          <div className="topbar-actions">
            <button
              type="button"
              className="round-button only-wide"
              aria-label="Help"
            >
              <HelpIcon />
            </button>
            <button type="button" className="round-button" aria-label="Notifications">
              <BellIcon />
              <span className="notify-dot" />
            </button>
            <button
              type="button"
              className="round-button only-wide"
              data-plain="true"
              aria-label="Assistant"
            >
              <SparkleIcon size={20} />
            </button>
            <button type="button" className="user-button">
              <span className="avatar">
                <Image src="/brand/avatar.png" alt="" width={32} height={32} />
              </span>
              <span className="user-name">Madhur Rastogi</span>
              <span className="user-chevron nav-icon">
                <ChevronDownIcon size={20} />
              </span>
            </button>
          </div>
        </header>

        {children}
      </div>
    </div>
  );
}
