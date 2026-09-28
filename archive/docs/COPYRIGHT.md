# © Copyright & Legal Protection

Protect your Word Bocce intellectual property and comply with relevant laws.

---

## Copyright Protection

### 1. Copyright Notice

**Add to all source files**:

```python
"""
Word Bocce - A multiplayer word vector game
Copyright (c) 2025 [Your Name/Company]
All rights reserved.
"""
```

**Add to website footer** (index.html):

```html
<footer style="text-align: center; padding: 20px; color: #999; font-size: 0.85em;">
    <p>© 2025 [Your Name/Company]. All rights reserved.</p>
    <p><a href="/terms">Terms of Service</a> | <a href="/privacy">Privacy Policy</a></p>
</footer>
```

### 2. Choose a License

**Options**:

**Proprietary (Closed Source)**:
- Full control over distribution
- Can monetize without restrictions
- No requirement to share source
- **Recommended** if you plan to sell commercially

**Open Source (GPL-3.0)**:
- Allows others to use and modify
- Requires derivative works to be open source
- Good for community building
- Harder to monetize directly

**Open Source (MIT)**:
- Very permissive
- Allows commercial use by others
- Only requires attribution
- Easy to fork and modify

**Recommendation**: Start with **Proprietary** until you determine monetization strategy.

Create LICENSE file:

```
Copyright (c) 2025 [Your Name]

All rights reserved.

This software and associated documentation files (the "Software") may not be
copied, modified, merged, published, distributed, sublicensed, or sold without
prior written permission from the copyright holder.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

### 3. Register Copyright (Optional but Recommended)

**US Copyright Registration**:

1. Go to https://www.copyright.gov/registration/
2. Register as "Computer Program" or "Website"
3. Cost: $65 (online) or $125 (paper)
4. Benefits:
   - Public record of ownership
   - Can sue for infringement
   - Statutory damages available
   - Stronger legal position

**Timeline**: Register within 3 months of publication for maximum protection.

### 4. Trademark "Word Bocce" Name

**Why Trademark**:
- Prevents others from using "Word Bocce" name
- Protects brand identity
- Increases value if selling

**Steps**:

1. **Search existing trademarks**: https://www.uspto.gov/trademarks
   - Check if "Word Bocce" is available
   - Search similar names

2. **File trademark application**:
   - DIY: $250-$350 per class (https://www.uspto.gov/)
   - Attorney: $1000-$2000 total
   - Class 41: "Entertainment services, namely, providing an online game"
   - Class 9: "Downloadable game software" (if applicable)

3. **Timeline**: 8-12 months for approval

4. **Maintenance**: Renew every 10 years

**Alternative**: Use ™ symbol without registration (common law trademark)

### 5. Protect Game Mechanics

**Problem**: Game mechanics are generally not copyrightable.

**Solutions**:

**Design Patent** (expensive, ~$5k-15k):
- Protects unique visual design
- Lasts 15 years
- Example: Protect puzzle UI layout

**Trade Secret**:
- Keep embedding model configuration secret
- Non-disclosure agreements with collaborators
- Document creation dates

**First-Mover Advantage**:
- Build brand recognition quickly
- Establish as "original" Word Bocce
- Community building

---

## Legal Compliance

### 6. Privacy Policy (Required)

**Why**: Required by law if collecting any user data.

**What to include**:

```markdown
# Privacy Policy

Last updated: [Date]

## Information We Collect
- Player names (temporary, not stored permanently)
- IP addresses (for rate limiting only)
- Game statistics (puzzle completion, scores)
- Cookies (for session management)

## How We Use Information
- To provide game functionality
- To prevent abuse and spam
- To improve game experience
- We do NOT sell or share your data with third parties

## Data Retention
- Match data: Deleted after 24 hours
- Puzzle progress: Kept indefinitely (anonymous)
- IP logs: Kept for 30 days

## Your Rights
- Request data deletion: [email]
- Opt-out of analytics: [instructions]

## Cookies
We use essential cookies only for game sessions.
No tracking or advertising cookies.

## Children's Privacy
Our game is not directed at children under 13.
We do not knowingly collect data from children.

## Contact
Questions: [your-email]
```

**Generator tools**:
- https://www.privacypolicygenerator.info/
- https://www.termsfeed.com/privacy-policy-generator/

### 7. Terms of Service (Required)

```markdown
# Terms of Service

## Acceptance
By using Word Bocce, you agree to these terms.

## Use License
We grant you a limited, non-exclusive, non-transferable license
to use Word Bocce for personal entertainment.

## Prohibited Conduct
- No cheating, hacking, or exploits
- No offensive player names
- No automated bots or scripts
- No commercial use without permission

## User Content
Player names and game moves are not considered creative works.
We may display leaderboards and statistics publicly.

## Disclaimer
THE GAME IS PROVIDED "AS IS" WITHOUT WARRANTY.
WE ARE NOT LIABLE FOR ANY DAMAGES FROM USE.

## Changes
We may update these terms at any time.
Continued use constitutes acceptance.

## Governing Law
These terms are governed by [Your State/Country] law.

## Contact
[Your email]
```

### 8. GDPR Compliance (If EU Users)

**Requirements**:

1. **Consent**: Get explicit consent before collecting data
2. **Right to Access**: Users can request their data
3. **Right to Deletion**: Users can request data deletion
4. **Data Portability**: Users can export their data
5. **Breach Notification**: Report breaches within 72 hours

**Implementation**:

```html
<!-- Cookie consent banner -->
<div id="cookie-banner" style="position: fixed; bottom: 0; width: 100%; background: #333; color: white; padding: 15px; text-align: center;">
    We use essential cookies for game functionality.
    <button onclick="acceptCookies()">Accept</button>
    <a href="/privacy">Learn more</a>
</div>
```

**Minimal GDPR approach**: Don't store personal data
- Use anonymous IDs instead of names
- Don't track users
- Clear data after 24 hours

### 9. DMCA Takedown Protection

**If users can input content** (like joker cards):

Add DMCA agent:

```markdown
## DMCA Notice

If you believe content infringes your copyright:

Send notice to: [your-email]

Include:
- Your contact information
- Description of copyrighted work
- Location of infringing content
- Statement of good faith
- Statement of accuracy
- Your signature
```

**Register DMCA agent**: https://www.copyright.gov/dmca-directory/

**Cost**: $6 filing fee

### 10. Disclaimers

**Add to About page**:

```markdown
## Disclaimers

Word Bocce uses pre-trained word embeddings (GloVe) which
may contain biases present in training data. We do not
endorse or control word relationships in the vector space.

The game is for entertainment purposes only. Educational
claims are not medical or professional advice.

Some word combinations may be inappropriate. We filter
common profanity but cannot guarantee all combinations
will be appropriate.
```

---

## Third-Party Licenses

### 11. Attribute Dependencies

**Check licenses** of dependencies:

```bash
pip install pip-licenses
pip-licenses --format=markdown --output-file=LICENSES.md
```

**Common licenses**:
- **MIT**: Very permissive, just need attribution
- **Apache 2.0**: Permissive, need attribution + patent grant
- **GPL**: Requires your code to be GPL (avoid if closed source)

**GloVe embeddings**: Apache 2.0 License
- Must include attribution
- Add to README:

```markdown
## Attributions

This project uses GloVe word embeddings:
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
GloVe: Global Vectors for Word Representation.
https://nlp.stanford.edu/projects/glove/

Licensed under Apache License 2.0.
```

**FastAPI**: MIT License
- Include in attribution

### 12. Create ATTRIBUTIONS.md

```markdown
# Third-Party Licenses

## GloVe Word Embeddings
- **License**: Apache 2.0
- **Source**: https://nlp.stanford.edu/projects/glove/
- **Citation**:
  Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
  GloVe: Global Vectors for Word Representation.

## FastAPI
- **License**: MIT
- **Source**: https://github.com/tiangolo/fastapi
- **Copyright**: © 2018 Sebastián Ramírez

## NumPy
- **License**: BSD 3-Clause
- **Source**: https://github.com/numpy/numpy

## Gensim
- **License**: LGPL 2.1
- **Source**: https://github.com/RaRe-Technologies/gensim

For a complete list, see LICENSES.md
```

---

## Domain & Hosting

### 13. Domain Name

**Register domain**: wordbocce.com

**Registrars**:
- Namecheap: ~$10/year
- Google Domains: ~$12/year
- Cloudflare: ~$10/year (+ free DDoS protection)

**Also consider**:
- wordbocce.app
- wordbocce.io
- wordbocce.gg

### 14. Defensive Registrations

**Protect brand**:
- wordbocce.net
- wordbocce.org
- word-bocce.com

**Cost**: ~$10/year each

**Alternative**: Monitor with Google Alerts for "word bocce"

---

## Monetization Considerations

### 15. Revenue Models

**Free with Ads**:
- Requires privacy policy updates
- Need ad platform compliance (Google AdSense, etc.)
- GDPR/CCPA implications

**Freemium**:
- Free basic game
- Paid features: More puzzles, custom avatars, ad-free
- Needs payment processing compliance

**One-Time Purchase**:
- $2.99-$9.99 one-time fee
- Simpler legally than subscriptions

**Important**: Consult lawyer for paid versions

### 16. Tax Implications

**If monetizing**:
- Register business entity (LLC recommended)
- Get EIN from IRS
- Collect sales tax (varies by state/country)
- Report revenue on taxes

**Recommendation**: Consult accountant if revenue > $1000/year

---

## Checklist

### Immediate (Before Public Launch)
- [ ] Add copyright notices to code
- [ ] Create LICENSE file
- [ ] Write Privacy Policy
- [ ] Write Terms of Service
- [ ] Create ATTRIBUTIONS.md
- [ ] Add disclaimers
- [ ] Register domain name

### Within 3 Months
- [ ] Register copyright with USPTO
- [ ] File trademark application
- [ ] Set up DMCA agent (if applicable)
- [ ] Review GDPR compliance

### Before Monetization
- [ ] Consult lawyer on commercial terms
- [ ] Set up business entity
- [ ] Register for tax collection
- [ ] Update privacy policy for payments

### Ongoing
- [ ] Monitor for trademark infringement
- [ ] Review privacy policy annually
- [ ] Update attributions when dependencies change

---

## Resources

### Legal
- **Copyright Registration**: https://www.copyright.gov/
- **Trademark Search**: https://www.uspto.gov/
- **DMCA Agent**: https://www.copyright.gov/dmca-directory/

### Templates
- **Privacy Policy**: https://www.termsfeed.com/
- **Terms Generator**: https://www.termsandconditionsgenerator.com/
- **License Chooser**: https://choosealicense.com/

### Consultation
- **LegalZoom**: $300-500 for basic review
- **UpCounsel**: $200-400/hr for IP lawyer
- **SCORE**: Free business mentoring (https://www.score.org/)

---

## Disclaimer

**This is not legal advice.** Consult a licensed attorney for your specific situation.
Different jurisdictions have different requirements.

**When to get a lawyer**:
- Before accepting investment
- Before monetizing significantly (>$10k/year)
- If you receive cease & desist
- For trademark/patent filings
- For complex licensing arrangements

**Cost**: $200-400/hour for IP attorney. Budget $1000-2000 for basic setup.

---

## Contact

For legal questions specific to your situation, consult:
- IP attorney in your jurisdiction
- Your local bar association for referrals
- Online legal services (LegalZoom, Rocket Lawyer)
