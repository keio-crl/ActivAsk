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

The videos are provided as real-robot execution examples. The CSV analysis and minimal examples do not require the video files.

## Failure Case Notes

The failure videos illustrate representative physical failure modes observed during robot execution:

- `Bottle/Failed/Failed_bottle_01.mp4` and `Bottle/Failed/Failed_bottle_02.mp4`: bottle grasps that failed because the bottle surface was smooth and difficult for the gripper to secure.
- `Cup/Failed/Failed_cup_01.mp4` and `Cup/Failed/Failed_cup_02.mp4`: cup grasps that failed because the selected grasp point was offset from the actual object contact location.
- `Utensil/Failed/Failed_utensil_01.mp4`: a fork grasp that failed because the fork head rebounded during contact.
- `Utensil/Failed/Failed_utensil_02.mp4`: a spoon grasp that failed because the grasp point was too close to the edge of the spoon head.
