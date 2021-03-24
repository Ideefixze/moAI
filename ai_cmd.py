import os
import secret
import pandas as pd
import re 
from unidecode import unidecode

import discord
from discord.ext import commands

import model_trainer as mt
import json

import messages

def check_me(user_id):
    return user_id==secret.ADMIN

def format_text(text):
    text = text.lower()
    text = re.sub(r"\:<^()>*\:", "", text)  
    text = ''.join([i for i in text if i.isalpha() or i in ['.',',',' ','/','-','—','\n','\r']])     
    text = unidecode(text)
    return text

def is_viable (msg):
    if len(msg.content) <= 75 and not msg.author.bot:
        return True
    else:
        return False

class AICommands(commands.Cog):

    def __init__(self, bot):
        self.bot = bot
    
        mfile = open("meta.json", "r")
        self.s_dict = json.loads(mfile.read())
        mfile.close()

        self.model = mt.get_model((0,mt.seq_len,1),len(self.s_dict))
        

    @commands.command(name='reload_model')
    async def reload_model_command(self, ctx, model_filename='model'):
        if check_me(ctx.author.id):
            await ctx.channel.send(messages.RELOAD_START_MSG)
        else:
            await ctx.channel.send(messages.NO_ACCES_MSG)
            return None

        self.model.load_weights("models/"+model_filename+".hdf5")

        await ctx.channel.send(messages.FINISHED_MSG)

    @commands.command(name='oracle')
    async def oracle_command(self, ctx, lng, temperature, *begin):
        pattern = " ".join(begin[:])

        pattern = format_text(pattern) + " "
        pattern = pattern[-mt.seq_len:]

        prediction = mt.predict(self.model,self.s_dict,pattern,int(lng),float(temperature))

        await ctx.channel.send(prediction + " " + messages.COOL_EMOJI)


    @commands.command(name='get_channel')
    async def get_channel(self, ctx, filename='data'):
        data = pd.DataFrame(columns=['content'])
        
        if check_me(ctx.author.id):
            await ctx.channel.send(messages.GET_START_MSG)
        else:
            await ctx.channel.send(messages.NO_ACCES_MSG)
            return None
        try:
            async for msg in ctx.channel.history(limit=50000):     
                text = format_text(msg.content)
                if is_viable(msg):
                    data = data.append({'content': text}, ignore_index=True)
        except:
            print("Error occured!")
            return None       

        await ctx.channel.send(messages.FINISHED_MSG)
        data = data.reindex(index=data.index[::-1])    
        file_location = f"data/{filename}.csv"
        data.to_csv(file_location)


    @commands.command(name='get_all')
    async def get_all_command(self, ctx, filename='data'):

        data = pd.DataFrame(columns=['content'])
        
        if check_me(ctx.author.id):
            await ctx.channel.send(messages.GET_START_MSG)
        else:
            await ctx.channel.send(messages.NO_ACCES_MSG)
            return None

        for channel in ctx.guild.text_channels:
            print(channel.name)
            try:
                async for msg in channel.history(limit=2500):     
                    text = format_text(msg.content)
                    if is_viable(msg):
                        data = data.append({'content': text}, ignore_index=True)
            except:
                print("Error occured!")     

        await ctx.channel.send(messages.FINISHED_MSG)
        data = data.reindex(index=data.index[::-1])    
        file_location = f"data/{filename}.csv"
        data.to_csv(file_location)