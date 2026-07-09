export const SPEAKER_THEMES = [
  { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.4)', text: '#3b82f6', primary: '#3b82f6' }, // Leader (Blue)
  { bg: 'rgba(20, 184, 166, 0.15)', border: 'rgba(20, 184, 166, 0.4)', text: '#14b8a6', primary: '#14b8a6' }, // HR (Teal)
  { bg: 'rgba(168, 85, 247, 0.15)', border: 'rgba(168, 85, 247, 0.4)', text: '#a855f7', primary: '#a855f7' }, // Junior (Purple)
  { bg: 'rgba(156, 163, 175, 0.15)', border: 'rgba(156, 163, 175, 0.4)', text: '#9ca3af', primary: '#9ca3af' }, // Other (Gray)
  { bg: 'rgba(0, 240, 255, 0.15)', border: 'rgba(0, 240, 255, 0.4)', text: '#00F0FF', primary: '#00F0FF' }, // Accent Cyan backup
  { bg: 'rgba(249, 115, 22, 0.15)', border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', primary: '#f97316' }, // Orange backup
];

export const getSpeakerTheme = (speaker, speakersList = []) => {
  const list = Array.isArray(speakersList) ? speakersList : [];
  const idx = list.indexOf(speaker);
  return SPEAKER_THEMES[idx >= 0 ? idx % SPEAKER_THEMES.length : 0];
};

export const mapRoleLabel = (role) => {
  if (typeof role !== 'string' || !role) return 'Other';
  const roleLower = role.trim().toLowerCase();
  if (roleLower === 'manager' || roleLower === 'leader') {
    return 'Leader';
  }
  if (roleLower === 'hr') {
    return 'HR';
  }
  if (roleLower === 'junior') {
    return 'Junior';
  }
  return 'Other';
};

export const capitalize = (str) => {
  if (typeof str !== 'string' || !str) return '';
  const mapped = mapRoleLabel(str);
  if (mapped === 'hr' || mapped === 'HR') return 'HR';
  return mapped;
};
