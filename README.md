# Pylon: NFL anytime TD model

Pylon prices six NFL player prop markets each week from nflverse data: anytime TD, rushing TDs, passing TDs, rushing yards, receiving yards, and passing yards. Every market is trained on one season and tested blind on the next.
It rebuilds itself automatically on GitHub and publishes to a free GitHub Pages site.

## One-time setup (about 5 minutes)

1. Create a new GitHub repository and upload everything in this folder (including the hidden `.github` folder).
2. In the repository, open **Settings → Pages**. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open the **Actions** tab, choose **Rebuild Pylon**, and click **Run workflow**.
   The first run takes 5–10 minutes because it downloads four seasons of play-by-play.
4. When it finishes, your site is live at `https://<your-username>.github.io/<repo-name>/`.

After that it rebuilds every morning at 9 AM Eastern, plus an extra refresh at noon Eastern on Thursdays and Sundays.
The season and week are detected automatically: once a week's games have results, the site moves on to the next week.

## What each rebuild does

- Downloads the latest play-by-play, snap counts, rosters, injuries, depth charts, and Vegas lines from nflverse
- Re-trains the model on the two most recent completed seasons and scores the prior season blind as a backtest
- Prices every active skill player for the upcoming week and writes `site/index.html`

## Running it on your own computer

```
pip install -r requirements.txt
python pylon_build.py                      # auto-detects season and week
python pylon_build.py --season 2026 --week 4
```

Then open `site/index.html` in a browser.

## Notes

- Sportsbook anytime TD odds are not in nflverse. Enter or paste them in the app; they are saved in your browser.
- nflverse updates play-by-play within a day of games and injury reports during the week. Lines come from nflverse's schedule file.
- Red zone snap data (participation) is only published for completed seasons, so it is used as a prior-season feature.
