"""Reported results of Milani, Franzoni & Florindi (2026), Tables 4-7, for side-by-side comparison.

All values are percentages except dSI. "Defended" means the u1.X / u2.X condition, which is the
`ope` defense in this project. Gemini is listed as in the paper (2.5 Flash); this project
currently runs gemini-3.8-flash because 2.5 Flash is unavailable to new API users.
"""

# Table 4: global performance across 27 CVs and 4 injection types
TABLE4 = {  # model key: (paper label, ASR no defense, ASR with defense, ME, DAR)
    "gpt": ("ChatGPT (4.0)", 94.2, 12.5, 86.7, 91.0),
    "copilot": ("Copilot (GPT-4)", 91.8, 14.2, 84.5, 89.1),
    "grok": ("Grok 3.0", 93.1, 16.4, 82.4, 88.5),
    "gemini": ("Gemini (2.5 flash)", 92.5, 19.8, 78.6, 82.3),
    "perplexity": ("Perplexity", 95.4, 35.2, 63.1, 58.0),
    "deepseek": ("DeepSeek V3", 98.0, 45.5, 53.6, 50.2),
}

# Table 5: by seniority, aggregated across LLMs: (ASR defended, ME, DAR)
TABLE5 = {"Junior": (15.2, 84.8, 88.6), "Mid": (22.4, 77.6, 79.1), "Senior": (33.1, 66.9, 68.4)}

# Table 6: by achievement density, aggregated across LLMs: (avg AD, dSI, ME)
TABLE6 = {"High": (2.5, 1.2, 71.4), "Medium": (1.2, 1.9, 79.2), "Low": (0.4, 2.6, 86.5)}

# Table 7: by injection type: (ASR defended, ME, DAR)
TABLE7 = {"pos": (14.2, 85.8, 92.2), "neg": (8.6, 91.4, 96.1), "posneg": (22.8, 77.2, 88.4),
          "posmed": (32.2, 67.8, 85.5)}
