# JJP movie-production launch kit

Your movie-creation adaptation of the supplied **JJP Launch System**, including the existing Movie Upscaler. No notebooks or models were changed.

- [Read the complete guide](JJP_MOVIE_LAUNCH_SYSTEM.md)
- [Open the printable PDF](JJP_MOVIE_LAUNCH_SYSTEM.pdf)
- [Download the complete kit](https://github.com/bothashaun-maker/Movies/raw/refs/heads/main/docs/movie_launch/JJP_MOVIE_LAUNCH_KIT.zip)

**Start with Section 00:** make one 15-second scene from an approved reference, edit it to your song, then test upscaling. Section 03 tells you exactly which model packs to attach. Section 07 covers the Movie Upscaler's two workflows, settings, audio, pauses, recovery and saving.

## Reusable templates

Open CSV files in Excel, Google Sheets or LibreOffice. Copy a blank template for each project. Example shot rows are planning suggestions, not completed renders.

| File | Use |
| --- | --- |
| [Episode brief](templates/episode_brief.md) | One-page creative and technical plan |
| [Eight episode cards](templates/episode_cards.csv) | Worlds, cast, palette, ending and PDF hook windows |
| [Shot list](templates/shot_list.csv) | Three sample shots for the first pilot; extend with your own |
| [Render log](templates/render_log.csv) | Prompts, seeds, model provenance, timing and accepted takes |
| [Upscale log](templates/upscale_log.csv) | Source mapping, restoration settings, checks and backups |
| [Asset register](templates/asset_register.csv) | Sources, permissions and durable file locations |
| [Weekly plan](templates/weekly_plan.csv) | Two-hour active-work blocks; record GPU time separately |
| [Release tracker](templates/release_tracker.csv) | Readiness, actual links and measured results |
| [Release checklist](templates/release_checklist.md) | Picture, sound, scripture, rights and archive review |

Song timings are transcribed planning inputs from the supplied PDF; verify them against your audio. No new render, model/license audit or platform-policy verification was performed for this documentation delivery. Existing runtime limitations remain in the linked technical guides.

## Verification

The PDF has 18 pages. Local Markdown targets, all CSV row widths, the eight episode cards and their proposed timestamp ranges, expected PDF text and the 12-file ZIP integrity were checked. The cover, tool table and upscaler page were visually inspected. These are document checks, not model or GPU validation.

## Rebuild the downloadable documents

`build_pdf.py` reads the Markdown guide and templates. It needs Python, `reportlab==4.4.9` and DejaVu Sans system fonts. Run `python docs/movie_launch/build_pdf.py` from the repository root. It writes only the PDF and ZIP in this directory. The downloadable ZIP includes the guide, PDF and templates; its GitHub-relative Markdown links work when browsing the repository, while the PDF contains full GitHub links.
