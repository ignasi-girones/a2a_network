import ReactMarkdown from 'react-markdown';

interface Props {
  children: string;
  className?: string;
}

export function Markdown({ children, className = '' }: Props) {
  return (
    <div className={`prose prose-sm max-w-none prose-headings:text-gray-800 prose-headings:text-sm prose-headings:mt-3 prose-headings:mb-1 prose-p:text-xs prose-p:leading-relaxed prose-p:my-1 prose-li:text-xs prose-li:my-0 prose-ul:my-1 prose-ol:my-1 prose-strong:text-gray-800 prose-code:text-[11px] prose-code:bg-gray-100 prose-code:px-1 prose-code:rounded prose-pre:bg-gray-100 prose-pre:text-[11px] prose-blockquote:text-xs prose-blockquote:border-gray-300 prose-blockquote:not-italic ${className}`}>
      <ReactMarkdown>{children}</ReactMarkdown>
    </div>
  );
}
