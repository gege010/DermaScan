import { createContext, useContext, useEffect, useState } from 'react';
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { Assessment } from './types';
import type { JournalEntry } from './types';
import { JOURNAL_KEY, readEntries, storageEnabled } from './storage';
export type ScanDraft = {
  file: File | null;
  symptoms: string[];
  area: string;
  duration: string;
  imageKind: string;
  consent: boolean;
};
const initialDraft: ScanDraft = {
  file: null,
  symptoms: [],
  area: 'face',
  duration: 'unknown',
  imageKind: 'clinical',
  consent: false,
};
type Session = {
  draft: ScanDraft;
  setDraft: Dispatch<SetStateAction<ScanDraft>>;
  result: Assessment | null;
  setResult: Dispatch<SetStateAction<Assessment | null>>;
  journalDraft: string;
  setJournalDraft: Dispatch<SetStateAction<string>>;
  entries: JournalEntry[];
  setEntries: Dispatch<SetStateAction<JournalEntry[]>>;
  persistent: boolean;
  setPersistent: Dispatch<SetStateAction<boolean>>;
  journalEditor: { area: string; feeling: string; editing: string | null };
  setJournalEditor: Dispatch<
    SetStateAction<{ area: string; feeling: string; editing: string | null }>
  >;
};
const Context = createContext<Session | null>(null);
export function SessionProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState(initialDraft);
  const [result, setResult] = useState<Assessment | null>(null);
  const [journalDraft, setJournalDraft] = useState('');
  const [entries, setEntries] = useState<JournalEntry[]>(readEntries);
  const [persistent, setPersistent] = useState(storageEnabled);
  const [journalEditor, setJournalEditor] = useState<{
    area: string;
    feeling: string;
    editing: string | null;
  }>({ area: 'Wajah', feeling: 'Belum yakin', editing: null });
  useEffect(() => {
    function update(event: StorageEvent) {
      if (event.key === JOURNAL_KEY || event.key === null) {
        setEntries(readEntries());
        setPersistent(storageEnabled());
      }
    }
    window.addEventListener('storage', update);
    return () => window.removeEventListener('storage', update);
  }, []);
  useEffect(() => {
    if (!journalDraft.trim()) return;
    function guard(e: BeforeUnloadEvent) {
      e.preventDefault();
      e.returnValue = '';
    }
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [journalDraft]);
  return (
    <Context.Provider
      value={{
        draft,
        setDraft,
        result,
        setResult,
        journalDraft,
        setJournalDraft,
        entries,
        setEntries,
        persistent,
        setPersistent,
        journalEditor,
        setJournalEditor,
      }}
    >
      {children}
    </Context.Provider>
  );
}
export function useSession() {
  const ctx = useContext(Context);
  if (!ctx) throw new Error('Session provider missing');
  return ctx;
}
