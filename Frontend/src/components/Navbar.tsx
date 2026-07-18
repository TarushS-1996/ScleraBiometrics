import { useState } from 'react';
import SettingsDrawer from './SettingsDrawer';
import { SettingsIcon } from 'lucide-react';

interface NavbarProps {
    activeTab: 'main' | 'record';
    onTabChange: (tab: 'main' | 'record') => void;
}

export default function Navbar({ activeTab, onTabChange }: NavbarProps) {
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);

    const handleTabClick = (tab: 'main' | 'record') => {
        onTabChange(tab);
    };

    return (
        <>
            <nav className="flex items-center justify-between px-6 h-14 bg-bg-secondary border-b border-border-custom shrink-0 transition-all duration-300 ease-in-out sm:h-auto  sm:px-4 sm:flex-wrap sm:gap-3">
                <div className="flex items-center gap-2.5">
                    <img src="logo.svg" alt="Invincible Biometrics" className="w-[100px] h-[100px] text-accent-cyan" />
                </div>

                <div className='flex item-center'>
                    <div className="flex items-center gap-1">
                        <button
                            className={`flex items-center gap-2 px-5 py-2 border-none rounded-md text-[13px] font-medium cursor-pointer transition-all duration-200 ease-in-out hover:bg-white/5 ${activeTab === 'main' ? 'bg-accent-blue text-white' : 'bg-transparent text-text-secondary'}`}
                            onClick={() => handleTabClick('main')}
                        >
                            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M3 3h8v8H3V3zm0 10h8v8H3v-8zm10-10h8v8h-8V3zm0 10h8v8h-8v-8z" />
                            </svg>
                            Main Screen
                        </button>
                        <button
                            className={`flex items-center gap-2 px-5 py-2 border-none rounded-md text-[13px] font-medium cursor-pointer transition-all duration-200 ease-in-out hover:bg-white/5 ${activeTab === 'record' ? 'bg-accent-blue text-white' : 'bg-transparent text-text-secondary'}`}
                            onClick={() => handleTabClick('record')}
                        >
                            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13zM6 20V4h5v7h7v9H6z" />
                            </svg>
                            Record Screen
                        </button>
                        <button
                            className="flex items-center justify-center w-9 h-9 border-none rounded-md bg-transparent text-text-secondary cursor-pointer ml-2 transition-all duration-200 ease-in-out hover:bg-white/5 hover:text-text-primary"
                            onClick={() => setIsSettingsOpen(true)}
                        >
                            <SettingsIcon />
                        </button>
                    </div>
                </div>
            </nav>

            <SettingsDrawer isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
        </>
    );
}
