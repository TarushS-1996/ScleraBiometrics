import { useRef, useState } from 'react';
import { Image as ImageIcon, SlidersHorizontal } from 'lucide-react';

interface ControlInterfaceProps {
    userId: string;
    firstName: string;
    lastName: string;
    eyeSide: string;
    onUserIdChange: (v: string) => void;
    onNameChange: (fn: string, ln: string) => void;
    onEyeSideChange: (v: string) => void;
    onImageUpload: (dataUrls: string[], files: File[]) => void;
    onStartScan: () => void;
    onRegister: (fn: string, ln: string) => void;
    hasImage: boolean;
    isScanning: boolean;
    isRegistering: boolean;
    mode: 'verify' | 'add';
    onModeChange: (m: 'verify' | 'add') => void;
}

export default function ControlInterface({
    eyeSide, onEyeSideChange,
    firstName, lastName, onNameChange,
    onImageUpload, onStartScan, onRegister,
    hasImage, isScanning, isRegistering,
    mode, onModeChange
}: ControlInterfaceProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        if (files.length > 0) {
            const dataUrls: string[] = [];
            let loadedCount = 0;

            files.forEach((file) => {
                const reader = new FileReader();
                reader.onload = (event) => {
                    dataUrls.push(event.target?.result as string);
                    loadedCount++;
                    if (loadedCount === files.length) {
                        onImageUpload(dataUrls, files);
                    }
                };
                reader.readAsDataURL(file);
            });
        }
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        const files = Array.from(e.dataTransfer.files || []);
        if (files.length > 0) {
            const dataUrls: string[] = [];
            let loadedCount = 0;

            files.forEach((file) => {
                const reader = new FileReader();
                reader.onload = (event) => {
                    dataUrls.push(event.target?.result as string);
                    loadedCount++;
                    if (loadedCount === files.length) {
                        onImageUpload(dataUrls, files);
                    }
                };
                reader.readAsDataURL(file);
            });
        }
    };

    return (
        <div className="bg-bg-secondary border border-border-custom rounded-xl p-5 flex flex-col min-h-fit lg:min-h-[320px]">
            {/* Header */}
            <div className="flex items-center justify-between mb-5 flex-wrap gap-4">
                <div className="flex items-center gap-3">
                    <SlidersHorizontal className="w-5 h-5 text-accent-cyan shrink-0" />
                    <h3 className="text-[16px] font-medium text-text-primary tracking-wide">Control Interface</h3>
                </div>

                {/* Toggle Group */}
                <div className="flex items-center bg-[#151515] border border-[#2a2a2a] rounded-lg p-1 shadow-inner">
                    <button
                        onClick={() => onModeChange('verify')}
                        className={`px-4 py-1.5 text-[13px] font-medium rounded-md transition-all duration-200 ${mode === 'verify' ? 'bg-accent-blue text-white shadow-sm' : 'text-text-muted hover:text-text-secondary'}`}
                    >
                        Verify User
                    </button>
                    <button
                        onClick={() => onModeChange('add')}
                        className={`px-4 py-1.5 text-[13px] font-medium rounded-md transition-all duration-200 ${mode === 'add' ? 'bg-accent-blue text-white shadow-sm' : 'text-text-muted hover:text-text-secondary'}`}
                    >
                        Add User
                    </button>
                </div>
            </div>

            <div className="flex-1 flex flex-col gap-4">
                {/* Inputs Row */}
                <div className="flex gap-4 flex-col sm:flex-row">
                    {mode === 'add' && (
                        <>
                            <input
                                type="text"
                                placeholder="First Name"
                                className="flex-1 px-4 py-3 bg-[#111] border border-accent-cyan rounded-md text-text-primary text-[14px] outline-none placeholder:text-text-muted focus:shadow-[0_0_8px_rgba(0,191,255,0.2)] transition-shadow"
                                value={firstName}
                                onChange={(e) => onNameChange(e.target.value, lastName)}
                            />
                            <input
                                type="text"
                                placeholder="Last Name"
                                className="flex-1 px-4 py-3 bg-[#111] border border-accent-cyan rounded-md text-text-primary text-[14px] outline-none placeholder:text-text-muted focus:shadow-[0_0_8px_rgba(0,191,255,0.2)] transition-shadow"
                                value={lastName}
                                onChange={(e) => onNameChange(firstName, e.target.value)}
                            />
                        </>
                    )}
                    <select
                        className="px-4 py-3 bg-[#111] border border-accent-cyan rounded-md text-text-primary text-[14px] outline-none focus:shadow-[0_0_8px_rgba(0,191,255,0.2)] transition-shadow cursor-pointer appearance-none"
                        value={eyeSide}
                        onChange={(e) => onEyeSideChange(e.target.value)}
                    >
                        <option value="right">Right Eye</option>
                        <option value="left">Left Eye</option>
                    </select>
                </div>

                {/* Drag and Drop Zone */}
                <div
                    className={`flex-1 border border-dashed border-accent-cyan rounded-md flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-200 min-h-[110px] text-text-muted text-[13px] hover:bg-accent-cyan/5 ${dragActive ? 'bg-accent-cyan/10 border-white' : 'bg-[#111]'}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        accept="image/*"
                        className="hidden"
                        multiple
                    />
                    <ImageIcon className="w-7 h-7 text-accent-cyan" strokeWidth={1.5} />
                    <span className="text-[14px] text-text-secondary">
                        drag an image here or <span className="text-accent-cyan underline cursor-pointer hover:text-white transition-colors">Click here to Upload</span>
                    </span>
                </div>

                {/* Buttons Row */}
                <div className="flex gap-4 flex-col sm:flex-row mt-1">
                    {mode === 'verify' && (
                        <button
                            className="flex-1 py-3 px-4 rounded-md text-[14px] font-medium cursor-pointer transition-all duration-200 flex items-center justify-center gap-2 bg-[#111] border border-accent-magenta text-text-secondary hover:text-white hover:bg-accent-magenta/10 hover:shadow-[0_0_12px_rgba(255,0,128,0.2)] disabled:opacity-40 disabled:cursor-not-allowed"
                            onClick={onStartScan}
                            disabled={!hasImage || isScanning}
                        >
                            {isScanning ? 'Scanning...' : 'Start Scan'}
                        </button>
                    )}
                    {mode === 'add' && (
                        <button
                            className="flex-1 py-3 px-4 rounded-md text-[14px] font-medium cursor-pointer transition-all duration-200 flex items-center justify-center gap-2 bg-[#111] border border-accent-blue text-text-secondary hover:text-white hover:bg-accent-blue/10 hover:shadow-[0_0_12px_rgba(0,30,255,0.2)] disabled:opacity-40 disabled:cursor-not-allowed"
                            onClick={() => onRegister(firstName, lastName)}
                            disabled={!hasImage || isRegistering}
                        >
                            {isRegistering ? 'Saving...' : 'Save'}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
