# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an LLM playground repository for experimentation. It uses Node.js with pnpm as the package manager.

## Commands

- **Install dependencies:** `pnpm install`
- **Run tests:** `pnpm test` (not yet configured)
- **Package manager:** pnpm 10.28.1 (specified in package.json `packageManager` field)

## Structure

- `logs/` — Claude Code hook event logs (session start, prompt submit, tool use)
