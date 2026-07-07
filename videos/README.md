# Supplementary Videos

Videos are organized by object category and execution outcome:

- `Bottle/Success/`
- `Bottle/Failed/`
- `Cup/Success/`
- `Cup/Failed/`
- `Utensil/Success/`
- `Utensil/Failed/`

Filename format:

```text
<Outcome>_<category>_<index>.mp4
```

Examples:

```text
Success_cup_01.mp4
Failed_utensil_02.mp4
```

The videos are provided as real-robot execution examples. They are shown at normal playback speed and are not sped up. The CSV analysis and minimal examples do not require the video files.

## Failure Case Notes

The failure videos illustrate representative physical failure modes observed during robot execution:

- Bottle failures:
  - `Bottle/Failed/Failed_bottle_01.mp4`: grasp failure caused by the smooth bottle surface.
  - `Bottle/Failed/Failed_bottle_02.mp4`: grasp failure caused by the smooth bottle surface.
- Cup failures:
  - `Cup/Failed/Failed_cup_01.mp4`: grasp failure caused by an offset between the selected grasp point and the actual object contact location.
  - `Cup/Failed/Failed_cup_02.mp4`: grasp failure caused by an offset between the selected grasp point and the actual object contact location.
- Utensil failures:
  - `Utensil/Failed/Failed_utensil_01.mp4`: fork grasp failure caused by rebound at the fork head during contact.
  - `Utensil/Failed/Failed_utensil_02.mp4`: spoon grasp failure caused by a grasp point too close to the edge of the spoon head.
