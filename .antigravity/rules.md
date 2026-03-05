# Environment Configuration

```

### 2. Manually Select the Interpreter in the IDE

If the code editor still shows "path cannot be resolved," point the IDE to the binary:

1.  Press **Cmd + Shift + P** to open the Command Palette.
2.  Type **"Python: Select Interpreter"** and select it.
3.  Choose **"Enter interpreter path..."** and then click **"Find..."**.
4.  Press **Cmd + Shift + G** in the file picker to open the "Go to Folder" box.
5.  Type `/usr/local/bin/python3` and click **Select Interpreter**.

### 3. Restart the Agent Service

After changes, the running agent needs to recognize the new path:

*   Open the Command Palette (**Cmd + Shift + P**) and run **"Antigravity: Restart Agent Service"**.
*   Alternatively, type `continue` or start a **new chat session**.

**Pro Tip:** If using virtual environments, add a rule to your `rules.md` to `source .venv/bin/activate` before any execution task.

Does the agent now **execute terminal commands** using that path, or does it still fail during the **environment check**?
