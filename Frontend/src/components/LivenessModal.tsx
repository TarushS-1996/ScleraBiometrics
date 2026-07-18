import { X, Info } from 'lucide-react';

interface LivenessModalProps {
    isOpen: boolean;
    status: 'good' | 'incorrect';
    title: string;
    message: string;
    onClose: () => void;
    onTryAgain: () => void;
    onAddNewUser: () => void;
}

export default function LivenessModal({
    isOpen,
    status,
    title,
    message,
    onClose,
    onTryAgain,
    onAddNewUser,
}: LivenessModalProps) {
    if (!isOpen) return null;

    const isGood = status === 'good';
    const statusColor = isGood ? 'text-accent-green' : 'text-red-500';
    const statusLabel = isGood ? 'Access Granted' : 'Access Denied';
    const statusBorder = isGood ? 'border-accent-green/30' : 'border-red-500/30';

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-[1px] animate-in fade-in duration-300">
            <div
                className="w-full max-w-[450px] bg-[#1a1a1a] border border-accent-cyan rounded-xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden scale-in-center"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-3">
                    <div className="flex items-center gap-3">
                        <div className="flex items-center justify-center">
                            <Info className="w-5 h-5 text-accent-cyan" />
                        </div>
                        <span className="text-white font-semibold tracking-wide text-lg">Error</span>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-white/10 rounded-md transition-colors group"
                    >
                        <X className="w-5 h-5 text-white/40 group-hover:text-white" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4">
                    <div className={`bg-black text-center p-6 mb-6 rounded-md border ${statusBorder}`}>
                        <div className="mb-6 flex justify-center">
                            <img src="/error.svg" alt="" />
                        </div>

                        <h2 className="text-3xl font-semibold text-white mb-2 tracking-tight">{title}</h2>
                        <h1 className={`text-4xl font-semibold mb-2 uppercase tracking-tight ${statusColor}`}>{statusLabel}</h1>

                        <p className="text-base font-semibold leading-relaxed mb-2">
                            {message}
                        </p>
                        <p className="text-sm text-text-muted">
                            Click on add new user or try again to verify.
                        </p>
                    </div>

                    {/* Buttons */}
                    <div className="flex flex-col sm:flex-row gap-4 w-full">
                        <button
                            onClick={onAddNewUser}
                            className="flex-1 py-3.5 px-6 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-bold text-[15px] transition-all active:scale-95 shadow-lg shadow-blue-600/20"
                        >
                            Add new user
                        </button>
                        <button
                            onClick={onTryAgain}
                            className="flex-1 py-3.5 px-6 rounded-lg bg-[#111] border border-accent-cyan/50 hover:bg-white/5 text-white font-bold text-[15px] transition-all active:scale-95"
                        >
                            Try Again
                        </button>
                    </div>
                </div>
            </div >
        </div >
    );
}
