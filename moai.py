import os
import secret

import asyncio

import pandas as pd
import re 
from unidecode import unidecode

import discord
from discord.ext import commands

import ai_cmd

if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('outs'):
    os.makedirs('outs')
if not os.path.exists('models'):
    os.makedirs('models')

bot = commands.Bot(command_prefix="!!")
bot.add_cog(ai_cmd.AICommands(bot))

bot.run(secret.TOKEN)
