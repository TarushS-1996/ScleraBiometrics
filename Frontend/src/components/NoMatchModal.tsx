import { X, AlertCircle } from 'lucide-react';

interface NoMatchModalProps {
    isOpen: boolean;
    onClose: () => void;
    onTryAgain: () => void;
    onAddNewUser: () => void;
}

export default function NoMatchModal({ isOpen, onClose, onTryAgain, onAddNewUser }: NoMatchModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[3000] flex items-center justify-center p-4 bg-black/20 backdrop-blur-xs animate-in fade-in duration-200 ">
            <div className="bg-[#1a1b1e] border border-[#00BFFF] rounded-xl w-full max-w-md shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3">
                    <div className="flex items-center gap-2">
                        <AlertCircle className="w-5 h-5 text-[#00BFFF]" />
                        <span className="text-sm font-medium text-white">Error</span>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-white/10 rounded-md transition-colors text-text-muted"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4 m-4 flex flex-col items-center bg-gray-900">
                    <div className="mb-6">
                        <img src="/error.svg" alt="" />
                    </div>

                    <h2 className="text-3xl font-bold text-white text-center mb-1">Eye Not Matched</h2>
                    <h3 className="text-4xl font-extrabold text-[#FF3B30] text-center mb-4 tracking-tight">Access Denied</h3>

                    <p className="text-text-secondary text-sm text-center max-w-[280px]">
                        Click on add new user or try again to verify
                    </p>
                </div>

                {/* Footer/Buttons */}
                <div className="px-6 pb-8 flex gap-3">
                    <button
                        onClick={onAddNewUser}
                        className="flex-1 bg-[#001EFF] hover:bg-[#0019D9] text-white py-3 rounded-lg font-semibold transition-colors shadow-lg active:scale-[0.98]"
                    >
                        Add new user
                    </button>
                    <button
                        onClick={onTryAgain}
                        className="flex-1 bg-transparent border border-[#00BFFF] text-[#00BFFF] py-3 rounded-lg font-semibold hover:bg-[#00BFFF]/10 transition-colors active:scale-[0.98]"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        </div>
    );
}
