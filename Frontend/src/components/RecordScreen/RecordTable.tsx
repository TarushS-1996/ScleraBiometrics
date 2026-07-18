import { Image as ImageIcon, Trash2, ArrowUpDown, Download } from 'lucide-react';
import type { LogEntry } from '../../types';

interface RecordTableProps {
  data: LogEntry[];
  selectedRows: Set<number>;
  toggleRow: (index: number) => void;
  toggleAll: () => void;
  onPreview: (index: number) => void;
  onDownload: () => void;
  onDelete: (recordId: string) => void;
}

export default function RecordTable({ data, selectedRows, toggleRow, toggleAll, onPreview, onDownload, onDelete }: RecordTableProps) {
  return (
    <div className="flex-1 overflow-auto">
      <table className="w-full text-left border-collapse min-w-[1000px]">
        {/* ... existing thead ... */}
        <thead className="sticky top-0 bg-bg-secondary z-10 border-b border-border-custom shadow-sm">
          <tr>
            <th className="px-5 py-4 w-12 text-center text-text-secondary">
              <input
                type="checkbox"
                className="w-[15px] h-[15px] accent-accent-cyan cursor-pointer border-border-custom rounded-sm bg-transparent"
                checked={selectedRows.size === data?.length && data.length > 0}
                onChange={toggleAll}
              />
            </th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap cursor-pointer hover:text-accent-cyan transition-colors group">
              <div className="flex items-center gap-1.5 font-semibold">Sample ID <ArrowUpDown className="w-3.5 h-3.5 text-text-muted group-hover:text-accent-cyan" /></div>
            </th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap cursor-pointer hover:text-accent-cyan transition-colors group">
              <div className="flex items-center gap-1.5 font-semibold">User ID <ArrowUpDown className="w-3.5 h-3.5 text-text-muted group-hover:text-accent-cyan" /></div>
            </th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap cursor-pointer hover:text-accent-cyan transition-colors group">
              <div className="flex items-center gap-1.5 font-semibold">User Name <ArrowUpDown className="w-3.5 h-3.5 text-text-muted group-hover:text-accent-cyan" /></div>
            </th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap font-semibold">Action Type</th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap font-semibold">Result</th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap font-semibold">Liveliness</th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap font-semibold">Confidence</th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap cursor-pointer hover:text-accent-cyan transition-colors group">
              <div className="flex items-center gap-1.5 font-semibold">Timestamp <ArrowUpDown className="w-3.5 h-3.5 text-text-muted group-hover:text-accent-cyan" /></div>
            </th>
            <th className="px-4 py-4 text-[13px] font-medium text-text-primary whitespace-nowrap font-semibold">Option</th>
          </tr>
        </thead>
        <tbody>
          {data?.map((row, index) => {
            const isMatch = row.action === 'match';
            const sampleId = isMatch ? row.best_match_sample : row.sample;
            const userId = isMatch ? row.best_match_user_id : row.user_id;
            const actionType = isMatch ? 'Verify User' : 'Registered User';
            const resultLabel = isMatch ? (row.matched ? 'Matched' : 'Unmatched') : 'Registered';
            const confidence = isMatch ? `${(row.best_match_similarity * 100).toFixed(1)}%` : '-';
            const formattedDate = new Date(row.timestamp).toLocaleString(undefined, {
              day: '2-digit', month: 'short', year: 'numeric',
              hour: '2-digit', minute: '2-digit'
            });

            return (
              <tr
                key={index}
                className="border-b border-border-custom/50 hover:bg-white/[0.02] transition-colors"
              >
                <td className="px-5 py-4 w-12 text-center align-middle">
                  <input
                    type="checkbox"
                    className="w-[15px] h-[15px] accent-accent-cyan cursor-pointer rounded-sm border-border-custom"
                    checked={selectedRows.has(index)}
                    onChange={() => toggleRow(index)}
                  />
                </td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">{sampleId}</td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">{userId}</td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">{userId}</td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">{actionType}</td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <span className={`inline-flex items-center justify-center px-3 py-0.5 text-[12px] border rounded ${resultLabel === 'Matched' || resultLabel === 'Registered'
                    ? 'border-accent-green/30 text-accent-green'
                    : 'border-red-500/30 text-red-500'
                    }`}>
                    {resultLabel}
                  </span>
                </td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">Verified</td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">{confidence}</td>
                <td className="px-4 py-4 text-[13px] text-text-secondary whitespace-nowrap">
                  {formattedDate}
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-3.5 text-text-muted">
                    <button
                      onClick={() => onPreview(index)}
                      className="hover:text-text-primary transition-colors hover:scale-110 active:scale-95"
                    >
                      <ImageIcon className="w-[18px] h-[18px]" />
                    </button>
                    <button
                      onClick={onDownload}
                      className="hover:text-text-primary transition-colors hover:scale-110 active:scale-95"
                    >
                      <Download className="w-[18px] h-[18px]" />
                    </button>
                    <button
                      onClick={() => onDelete(row.record_id)}
                      className="hover:text-text-primary transition-colors hover:scale-110 active:scale-95"
                    >
                      <Trash2 className="w-[18px] h-[18px]" />
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
