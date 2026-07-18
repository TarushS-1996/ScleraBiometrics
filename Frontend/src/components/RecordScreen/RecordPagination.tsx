import { ChevronDown } from 'lucide-react';

interface RecordPaginationProps {
  totalItems: number;
  currentPage: number;
  onPageChange: (page: number) => void;
  itemsPerPage: number;
  onItemsPerPageChange: (count: number) => void;
}

export default function RecordPagination({ 
  totalItems, 
  currentPage, 
  onPageChange, 
  itemsPerPage, 
  onItemsPerPageChange 
}: RecordPaginationProps) {
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  const startIdx = (currentPage - 1) * itemsPerPage + 1;
  const endIdx = Math.min(currentPage * itemsPerPage, totalItems);
  return (
    <div className="flex items-center justify-between p-5 border-t border-border-custom bg-bg-secondary flex-wrap gap-4">
      <div className="flex items-center gap-3">
        <span className="text-[13px] text-text-secondary">Row Per Page</span>
        <div className="relative">
          <select 
            value={itemsPerPage}
            onChange={(e) => onItemsPerPageChange(Number(e.target.value))}
            className="appearance-none bg-bg-primary border border-border-custom text-accent-cyan text-[13px] rounded-md pl-3 pr-8 py-1.5 focus:outline-none focus:border-accent-cyan cursor-pointer"
          >
            <option value="10">10</option>
            <option value="20">20</option>
            <option value="50">50</option>
          </select>
          <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-accent-cyan pointer-events-none" />
        </div>
      </div>

      <div className="text-[13px] text-text-secondary hidden md:block">
        Showing {totalItems === 0 ? 0 : startIdx} to {endIdx} of {totalItems} entries
      </div>

      <div className="flex items-center text-[13px]">
        <button 
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
          disabled={currentPage === 1}
          className="px-3 py-1.5 text-text-secondary hover:text-text-primary transition-colors disabled:opacity-30"
        >
          Prev
        </button>
        <div className="flex items-center gap-1 px-2">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
            <button 
              key={page}
              onClick={() => onPageChange(page)}
              className={`w-[26px] h-[26px] flex items-center justify-center rounded font-medium shadow-sm transition-colors ${
                currentPage === page 
                  ? 'bg-accent-blue text-white' 
                  : 'text-text-secondary hover:bg-white/5 hover:text-text-primary'
              }`}
            >
              {page}
            </button>
          ))}
        </div>
        <button 
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage === totalPages || totalPages === 0}
          className="px-3 py-1.5 text-text-secondary hover:text-text-primary transition-colors disabled:opacity-30"
        >
          Next
        </button>
      </div>
    </div>
  );
}
