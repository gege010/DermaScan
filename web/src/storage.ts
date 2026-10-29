import type { JournalEntry } from './types';
export const JOURNAL_KEY = 'dermascan.journal.v1';
export function readEntries(): JournalEntry[] {
  try {
    const value = JSON.parse(localStorage.getItem(JOURNAL_KEY) || '[]');
    return Array.isArray(value)
      ? value
          .filter(
            (v): v is JournalEntry =>
              v &&
              typeof v.id === 'string' &&
              typeof v.date === 'string' &&
              Number.isFinite(Date.parse(v.date)) &&
              ['area', 'feeling', 'note'].every((k) => typeof v[k] === 'string') &&
              v.note.length <= 2000,
          )
          .slice(0, 100)
      : [];
  } catch {
    return [];
  }
}
export function storageEnabled() {
  try {
    return localStorage.getItem(JOURNAL_KEY) !== null;
  } catch {
    return false;
  }
}
