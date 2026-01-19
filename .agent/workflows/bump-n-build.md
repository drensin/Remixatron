---
description: Bump version numbers, create new tags, push to trigger build
---

1) discover the files in the source tree that contain the current app version number
2) tell me the current value and ask me for new value
3) write the new value to the necessary files
4) create a new tag for this app using that new version number. The format should be: v$version
5) commit the change, and push the new tags so that a new build is triggered by Git Actions

If you have any questions, ask me!