import { Search, ChevronDown, Download, Loader2 } from 'lucide-react';

interface RecordFiltersProps {
  onExport: () => void;
  exporting?: boolean;
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onReset: () => void;
  fromDate: string;
  onFromDateChange: (date: string) => void;
  toDate: string;
  onToDateChange: (date: string) => void;
  actionType: string;
  onActionTypeChange: (type: string) => void;
  onApplyFilters: () => void;
}

export default function RecordFilters({ 
  onExport, 
  exporting, 
  searchQuery, 
  onSearchQueryChange,
  onReset,
  fromDate,
  onFromDateChange,
  toDate,
  onToDateChange,
  actionType,
  onActionTypeChange,
  onApplyFilters
}: RecordFiltersProps) {
  return (
    <div className="flex items-center justify-between p-4 md:p-5 border-b border-border-custom gap-4 flex-wrap">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted group-focus-within:text-accent-cyan transition-colors" />
          <input 
            type="text" 
            placeholder="Search User ID..."
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            className="bg-bg-primary border border-border-custom text-text-primary text-[13px] rounded-md pl-9 pr-4 py-2 min-w-[220px] focus:outline-none focus:border-accent-cyan transition-all"
          />
        </div>

        <div className="relative">
          <input 
            type={fromDate ? "date" : "text"} 
            onFocus={(e) => (e.target.type = "date")}
            onBlur={(e) => !fromDate && (e.target.type = "text")}
            value={fromDate}
            onChange={(e) => onFromDateChange(e.target.value)}
            className="bg-bg-primary border border-border-custom text-accent-cyan text-[13px] rounded-md px-4 py-2 focus:outline-none focus:border-accent-cyan transition-all [color-scheme:dark] min-w-[140px]"
            placeholder="From Date"
          />
        </div>

        <div className="relative">
          <input 
            type={toDate ? "date" : "text"} 
            onFocus={(e) => (e.target.type = "date")}
            onBlur={(e) => !toDate && (e.target.type = "text")}
            value={toDate}
            onChange={(e) => onToDateChange(e.target.value)}
            className="bg-bg-primary border border-border-custom text-accent-cyan text-[13px] rounded-md px-4 py-2 focus:outline-none focus:border-accent-cyan transition-all [color-scheme:dark] min-w-[140px]"
            placeholder="To Date"
          />
        </div>

        <div className="relative">
          <select 
            value={actionType}
            onChange={(e) => onActionTypeChange(e.target.value)}
            className="appearance-none bg-bg-primary border border-border-custom text-accent-cyan text-[13px] rounded-md pl-4 pr-10 py-2 focus:outline-none focus:border-accent-cyan cursor-pointer transition-all"
          >
            <option value="">Action Type</option>
            <option value="verify">Verify User</option>
            <option value="register">Registered User</option>
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-accent-cyan pointer-events-none" />
        </div>

        <button 
          onClick={onReset}
          className="px-5 py-2 border border-red-500/50 text-red-500 text-[13px] rounded-md hover:bg-red-500/10 transition-colors"
        >
          Reset
        </button>

        <button 
          onClick={onApplyFilters}
          className="px-6 py-2 bg-accent-blue text-white text-[13px] font-medium rounded-md hover:bg-accent-blue/90 transition-colors"
        >
          Apply Filter
        </button>
      </div>

      <button 
        onClick={onExport}
        disabled={exporting}
        className="flex items-center gap-2 px-5 py-2 border border-accent-cyan text-accent-cyan text-[13px] rounded-md hover:bg-accent-cyan/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {exporting ? (
          <>
            Exporting...
            <Loader2 className="w-4 h-4 animate-spin" />
          </>
        ) : (
          <>
            Export
            <Download className="w-4 h-4" />
          </>
        )}
      </button>
    </div>
  );
}
