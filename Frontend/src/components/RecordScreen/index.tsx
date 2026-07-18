import { useState, useEffect, useRef, useCallback } from 'react';
import RecordFilters from './RecordFilters';
import RecordTable from './RecordTable';
import RecordPagination from './RecordPagination';
import { getLogs, exportLogs, fetchImage, deleteLog, type LogFilters } from '../../services/api';
import type { LogEntry } from '../../types';
import { X, AlertTriangle, ChevronLeft, ChevronRight } from 'lucide-react';

// Removed static DUMMY_DATA

export default function RecordScreen() {
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [totalItems, setTotalItems] = useState(0);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [previewIdx, setPreviewIdx] = useState<number | null>(null);
  const [previewLogs, setPreviewLogs] = useState<LogEntry[]>([]);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [tempSearch, setTempSearch] = useState('');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  const [actionType, setActionType] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const lastReqId = useRef(0);

  const fetchLogsData = useCallback(async () => {
    try {
      setLoading(true);
      const filters: LogFilters = {
        limit: itemsPerPage,
        offset: (currentPage - 1) * itemsPerPage,
        user_id: searchQuery || undefined,
        action: actionType === 'verify' ? 'match' : (actionType === 'register' ? 'new_user' : undefined),
        from_date: fromDate ? new Date(fromDate).toISOString() : undefined,
        to_date: toDate ? new Date(toDate).toISOString() : undefined
      };

      const response = await getLogs(filters);
      setLogs(response.logs);
      setTotalItems(response.total);
    } catch (err) {
      console.error('Error fetching logs:', err);
    } finally {
      setLoading(false);
    }
  }, [itemsPerPage, currentPage, actionType, fromDate, toDate, searchQuery]);

  useEffect(() => {
    fetchLogsData();
  }, [fetchLogsData, refreshTrigger]);


  const handleExport = async () => {
    try {
      setExporting(true);
      await exportLogs();
    } catch (err) {
      console.error('Error exporting logs:', err);
      // Optional: Add toast notification for error
    } finally {
      setExporting(false);
    }
  };

  const handleLogDelete = (recordId: string) => {
    setDeleteConfirmId(recordId);
  };

  const confirmDelete = async () => {
    if (!deleteConfirmId) return;

    try {
      setIsDeleting(true);
      await deleteLog(deleteConfirmId);
      setRefreshTrigger(prev => prev + 1);
      setDeleteConfirmId(null);
    } catch (err) {
      console.error('Error deleting log:', err);
      alert('Failed to delete log');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleViewImage = async (index: number, logsToUse: LogEntry[] = logs) => {
    const row = logsToUse[index];
    if (!row) return;

    const reqId = ++lastReqId.current;
    setPreviewIdx(index);
    setPreviewImage('loading');

    let currentLogs = logsToUse;
    let currentIndex = index;

    if (logsToUse === logs) {
      const isMatch = row.action === 'match';
      const userId = isMatch ? row.best_match_user_id : row.user_id;

      const sameUserLogs = logs.filter(l => {
        const lId = l.action === 'match' ? l.best_match_user_id : l.user_id;
        return lId === userId;
      });
      setPreviewLogs(sameUserLogs);
      currentIndex = sameUserLogs.findIndex(l => l.record_id === row.record_id);
      setPreviewIdx(currentIndex);
      currentLogs = sameUserLogs;
    }

    try {
      const currentRow = currentLogs[currentIndex];
      const isMatch = currentRow.action === 'match';
      const userId = isMatch ? currentRow.best_match_user_id : currentRow.user_id;
      const eyeSide = isMatch ? currentRow.best_match_eye_side : currentRow.eye_side;
      const sample = isMatch ? currentRow.best_match_sample : currentRow.sample;

      const url = await fetchImage(userId, eyeSide, sample);
      if (reqId === lastReqId.current) {
        setPreviewImage(url);
      }
    } catch (err) {
      console.error('Error fetching image:', err);
      if (reqId === lastReqId.current) {
        setPreviewImage('error');
      }
    }
  };

  const navigatePreview = (direction: 'prev' | 'next') => {
    if (previewIdx === null || previewLogs.length === 0) return;
    const newIdx = direction === 'prev' ? previewIdx - 1 : previewIdx + 1;
    if (newIdx >= 0 && newIdx < previewLogs.length) {
      handleViewImage(newIdx, previewLogs);
    }
  };

  const closePreview = () => {
    if (previewImage && previewImage.startsWith('blob:')) {
      window.URL.revokeObjectURL(previewImage);
    }
    setPreviewImage(null);
    setPreviewIdx(null);
    setPreviewLogs([]);
  };

  const toggleRow = (index: number) => {
    const newSet = new Set(selectedRows);
    if (newSet.has(index)) {
      newSet.delete(index);
    } else {
      newSet.add(index);
    }
    setSelectedRows(newSet);
  };

  const toggleAll = () => {
    if (selectedRows.size === logs.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(logs.map((_, i) => i)));
    }
  };

  const handleReset = () => {
    setSearchQuery('');
    setTempSearch('');
    setFromDate('');
    setToDate('');
    setActionType('');
    setCurrentPage(1);
  };

  const handleApplyFilters = () => {
    setSearchQuery(tempSearch);
    setCurrentPage(1);
  };

  return (
    <div className="flex-1 flex flex-col p-4 md:p-6 bg-bg-dark overflow-hidden font-inter">
      <div className="flex-1 bg-bg-secondary rounded-xl border border-border-custom flex flex-col overflow-hidden shadow-xl">
        <RecordFilters
          onExport={handleExport}
          exporting={exporting}
          searchQuery={tempSearch}
          onSearchQueryChange={setTempSearch}
          onReset={handleReset}
          fromDate={fromDate}
          onFromDateChange={setFromDate}
          toDate={toDate}
          onToDateChange={setToDate}
          actionType={actionType}
          onActionTypeChange={setActionType}
          onApplyFilters={handleApplyFilters}
        />

        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent-cyan"></div>
          </div>
        ) : (
          <RecordTable
            data={logs}
            selectedRows={selectedRows}
            toggleRow={toggleRow}
            toggleAll={toggleAll}
            onPreview={handleViewImage}
            onDownload={handleExport}
            onDelete={handleLogDelete}
          />
        )}

        <RecordPagination
          totalItems={totalItems}
          currentPage={currentPage}
          onPageChange={setCurrentPage}
          itemsPerPage={itemsPerPage}
          onItemsPerPageChange={(count) => {
            setItemsPerPage(count);
            setCurrentPage(1);
          }}
        />
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div
          className="fixed inset-0 z-60 flex items-center justify-center p-4 bg-black/20 backdrop-blur-sm animate-in fade-in duration-200"
          onClick={() => !isDeleting && setDeleteConfirmId(null)}
        >
          <div
            className="w-full max-w-md bg-bg-secondary rounded-xl border border-white/10 shadow-2xl overflow-hidden scale-in-center"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center shrink-0">
                  <AlertTriangle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Delete Log</h3>
                  <p className="text-text-muted mt-1">This action cannot be undone. Are you sure?</p>
                </div>
              </div>

              <div className="flex items-center gap-3 justify-end pt-4">
                <button
                  disabled={isDeleting}
                  onClick={() => setDeleteConfirmId(null)}
                  className="px-5 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-white font-medium transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  disabled={isDeleting}
                  onClick={confirmDelete}
                  className="px-5 py-2 rounded-lg bg-red-500 hover:bg-red-600 text-white font-semibold transition-all shadow-lg shadow-red-500/20 active:scale-95 disabled:opacity-50 flex items-center gap-2"
                >
                  {isDeleting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                      Deleting...
                    </>
                  ) : (
                    'Delete Record'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Image Preview Modal - Redesigned to match screenshot */}
      {previewIdx !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-300"
          onClick={closePreview}
        >
          <div
            className="flex flex-col w-[90%] max-w-[500px] bg-bg-secondary rounded-lg border-2 border-accent-cyan/40 overflow-hidden shadow-[0_0_40px_rgba(33,197,219,0.2)] scale-in-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 bg-white/5">
              <div className="flex items-center gap-3">
                <img src="/imageIcon.svg" alt="Eye Icon" />
                <h3 className="text-white font-semibold tracking-wide">Image Preview</h3>
              </div>
              <button
                onClick={closePreview}
                className="p-1.5 hover:bg-white/10 rounded-md transition-colors group"
              >
                <X className="w-5 h-5 text-white/60 group-hover:text-white" />
              </button>
            </div>

            {/* Body */}
            <div className="p-8 flex flex-col items-center">
              <div className="w-full aspect-square max-w-[320px] bg-black/40 rounded-xl border border-white/5 flex items-center justify-center overflow-hidden shadow-inner relative group">
                {previewImage === 'loading' ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-10 h-10 border-2 border-accent-cyan/20 border-t-accent-cyan rounded-full animate-spin"></div>
                    <p className="text-white/40 text-[13px] animate-pulse">Loading Scan...</p>
                  </div>
                ) : previewImage === 'error' ? (
                  <div className="flex flex-col items-center gap-3 text-red-400">
                    <p className="text-[13px] font-medium">Capture not found</p>
                  </div>
                ) : (
                  <img
                    src={previewImage || ''}
                    alt="Scan"
                    className="w-[90%] h-[90%] object-contain rounded-full shadow-2xl opacity-90 hover:opacity-100 transition-opacity"
                  />
                )}
              </div>
            </div>

            {/* Footer Navigation */}
            <div className="flex items-center justify-center gap-6 pb-8">
              <button
                onClick={() => navigatePreview('prev')}
                disabled={previewIdx === 0}
                className="p-2 bg-white/5 hover:bg-white/10 disabled:opacity-20 rounded-lg border border-white/10 transition-all active:scale-90"
              >
                <ChevronLeft className="w-5 h-5 text-white" />
              </button>

              <div className="text-white/80 font-medium text-[15px] min-w-[60px] text-center">
                <span className="text-white">{previewIdx + 1}</span>
                <span className="mx-1.5 text-white/20">of</span>
                <span className="text-white/40">{previewLogs.length}</span>
              </div>

              <button
                onClick={() => navigatePreview('next')}
                disabled={previewIdx === previewLogs.length - 1}
                className="p-2 bg-white/5 hover:bg-white/10 disabled:opacity-20 rounded-lg border border-white/10 transition-all active:scale-90"
              >
                <ChevronRight className="w-5 h-5 text-white" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
