# StudyBuddy — Frontend

> AI-powered study platform serving 500+ active users. Upload your notes and get instant AI-generated summaries, quizzes, flashcards, and concept maps.

🌐 **Live at [studybuddy.app](https://studybuddy-frontend-theta.vercel.app/)**

---

## What It Does

StudyBuddy lets students upload course materials and instantly generate study resources powered by AI. The platform supports full class management, syllabus parsing, calendar integration, and multiple study formats — all in a clean, responsive interface.

**Core features:**
- Upload PDFs and notes to get AI-generated summaries and study guides
- Generate quizzes with multiple choice questions from any document
- Flashcard generation for active recall practice
- Concept map visualization showing relationships between topics
- Syllabus parsing to extract and organize course structure
- iCalendar (.ics) import for assignment and exam tracking
- Class and document library management with delete functionality
- Freemium model with usage-gated AI generation
- Supabase authentication with persistent user sessions

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router) |
| UI | React 19, TypeScript |
| Styling | Tailwind CSS 4, Tailwind Typography |
| Auth & DB | Supabase (PostgreSQL + Auth) |
| Content Rendering | React Markdown, Marked, DOMPurify |
| Validation | Zod |
| Deployment | Vercel |

---

## Project Structure

```
/
├── components/
│   ├── AuthGate.tsx          # Authentication wrapper
│   ├── CalendarView.tsx      # Assignment/exam calendar
│   ├── ConceptMap.tsx        # Knowledge graph visualization
│   ├── DeleteButton.tsx      # Content deletion with confirmation
│   ├── FileDrop.tsx          # Drag and drop file upload
│   ├── FlashcardList.tsx     # Flashcard study interface
│   ├── ICalendarUploader.tsx # .ics file import for deadlines
│   ├── LandingPage.tsx       # Marketing landing page
│   ├── MarkdownView.tsx      # Rendered study guide display
│   ├── Nav.tsx               # Navigation with auth state
│   ├── QuizQuestion.tsx      # Interactive quiz component
│   ├── SideBar.tsx           # Class and document navigation
│   ├── StudyGuide.tsx        # AI summary display
│   └── SyllabusUploader.tsx  # Syllabus parsing interface
├── lib/
│   ├── api.ts                # Backend API client
│   ├── supabase.ts           # Supabase server client
│   ├── supabaseBrowser.ts    # Supabase browser client
│   ├── types.ts              # Shared TypeScript types
│   ├── delete.ts             # Delete operations
│   └── utils.ts              # Shared utilities
├── quiz/
│   ├── page.tsx              # Quiz list page
│   └── [id]/page.tsx         # Individual quiz page
└── public/                   # Static assets
```

---

## Architecture Highlights

**Authentication:** Supabase Auth with JWT sessions. AuthGate component wraps protected routes and redirects unauthenticated users. Separate server and browser Supabase clients handle SSR and client-side auth correctly.

**API Communication:** All AI generation requests route through the Python FastAPI backend. The `lib/api.ts` client handles request formatting, error handling, and auth header injection.

**Content Security:** DOMPurify sanitizes all AI-generated markdown before rendering to prevent XSS. `rehype-sanitize` provides additional sanitization in the React Markdown pipeline.

**Type Safety:** Zod schemas validate API responses at runtime. Shared TypeScript types in `lib/types.ts` ensure consistency across components.

---

## Local Development

### Prerequisites
- Node.js 18+
- Supabase project (for auth and database)
- StudyBuddy backend running locally (see [backend repo](https://github.com/MingLincs/studybuddy-backend))

### Setup

```bash
# Clone the repo
git clone https://github.com/MingLincs/studybuddy-frontend.git
cd studybuddy-frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# Add your Supabase URL, anon key, and backend API URL

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Environment Variables

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Related

- **Backend:** [studybuddy-backend](https://github.com/MingLincs/studybuddy-backend) — FastAPI + Python AI processing engine
- **Live App:** [studybuddy.app](https://studybuddy.app)
